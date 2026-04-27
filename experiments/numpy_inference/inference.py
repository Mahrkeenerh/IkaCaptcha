"""
NumPy-only CRNN inference. No torch, no onnxruntime.

Reuses the folded weights.bin from experiments/pure_python — same blob,
same manifest, just reshaped into ndarrays. Activations stay in (C, H, W)
layout for the CNN portion.

Public entry point:
    predict(x_chw, weights) -> str
where x_chw is a numpy array of shape (3, 48, 256), float32, in [-1, 1].
"""

import json
import struct
from pathlib import Path

import numpy as np

CHARSET = "abcdefghjklmnpqrstuvwxy23457"
BLANK = 0
NUM_CLASSES = len(CHARSET) + 1
IDX_TO_CHAR = {i + 1: c for i, c in enumerate(CHARSET)}

IMG_H = 48
IMG_W = 256

PP_DIR = Path(__file__).resolve().parent.parent / "pure_python"


# ---------------------------------------------------------------------------
# Weight loader — reads the same blob the pure-Python pass uses.
# ---------------------------------------------------------------------------

def load_weights(blob_path: str | Path = None, manifest_path: str | Path = None) -> dict:
    blob_path = Path(blob_path) if blob_path else PP_DIR / "weights.bin"
    manifest_path = Path(manifest_path) if manifest_path else PP_DIR / "weights_manifest.json"

    blob = blob_path.read_bytes()
    manifest = json.loads(manifest_path.read_text())

    out = {}
    for entry in manifest:
        n = entry["n"]
        shape = tuple(entry["shape"])
        arr = np.frombuffer(blob, dtype="<f4", count=n, offset=entry["offset"]).copy()
        out[entry["name"]] = arr.reshape(shape)
    return out


# ---------------------------------------------------------------------------
# Ops
# ---------------------------------------------------------------------------

def conv2d_3x3_pad1(x, w, b):
    """x: (C_in, H, W). w: (C_out, C_in, 3, 3). b: (C_out,). Output: (C_out, H, W)."""
    C_in, H, W = x.shape
    C_out = w.shape[0]
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    # im2col: rows are flattened 3x3xCin patches per output position.
    cols = np.empty((C_in * 9, H * W), dtype=x.dtype)
    idx = 0
    for kh in range(3):
        for kw in range(3):
            cols[idx * C_in:(idx + 1) * C_in] = xp[:, kh:kh + H, kw:kw + W].reshape(C_in, H * W)
            idx += 1
    w_mat = w.transpose(2, 3, 1, 0).reshape(9 * C_in, C_out).T  # (C_out, 9*C_in) matching col order
    return (w_mat @ cols).reshape(C_out, H, W) + b[:, None, None]


def conv2d_1x1(x, w, b):
    """x: (C_in, H, W). w: (C_out, C_in, 1, 1). b: (C_out,). Output: (C_out, H, W)."""
    C_in, H, W = x.shape
    C_out = w.shape[0]
    w_mat = w.reshape(C_out, C_in)
    return (w_mat @ x.reshape(C_in, H * W)).reshape(C_out, H, W) + b[:, None, None]


def relu(x):
    return np.maximum(x, 0.0, out=x)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def maxpool(x, kh, kw):
    """Stride = kernel. x: (C, H, W). Returns (C, H//kh, W//kw)."""
    C, H, W = x.shape
    H2, W2 = H // kh, W // kw
    return x[:, :H2 * kh, :W2 * kw].reshape(C, H2, kh, W2, kw).max(axis=(2, 4))


def se_block(x, w, prefix):
    fc1_w = w[f"{prefix}.se.fc1.weight"]
    fc1_b = w[f"{prefix}.se.fc1.bias"]
    fc2_w = w[f"{prefix}.se.fc2.weight"]
    fc2_b = w[f"{prefix}.se.fc2.bias"]
    pooled = x.mean(axis=(1, 2))                        # (C,)
    h1 = np.maximum(fc1_w @ pooled + fc1_b, 0.0)        # (mid,)
    scale = sigmoid(fc2_w @ h1 + fc2_b)                 # (C,)
    return x * scale[:, None, None]


def resblock(x, w, prefix, has_skip_conv):
    if has_skip_conv:
        residual = conv2d_1x1(x, w[f"{prefix}.skip.weight"], w[f"{prefix}.skip.bias"])
    else:
        residual = x

    out = conv2d_3x3_pad1(x, w[f"{prefix}.conv1.weight"], w[f"{prefix}.conv1.bias"])
    out = np.maximum(out, 0.0, out=out)
    out = conv2d_3x3_pad1(out, w[f"{prefix}.conv2.weight"], w[f"{prefix}.conv2.bias"])
    out = se_block(out, w, prefix)
    out = out + residual
    return np.maximum(out, 0.0, out=out)


def lstm_run(seq, w_ih, w_hh, b, hidden):
    """seq: (T, input_size). Returns (T, hidden)."""
    T, _ = seq.shape
    # Pre-project the input across all timesteps in one matmul.
    in_proj = seq @ w_ih.T + b  # (T, 4H)

    h = np.zeros(hidden, dtype=seq.dtype)
    c = np.zeros(hidden, dtype=seq.dtype)
    out = np.empty((T, hidden), dtype=seq.dtype)

    for t in range(T):
        gates = in_proj[t] + w_hh @ h
        i = sigmoid(gates[0:hidden])
        f = sigmoid(gates[hidden:2 * hidden])
        g = np.tanh(gates[2 * hidden:3 * hidden])
        o = sigmoid(gates[3 * hidden:4 * hidden])
        c = f * c + i * g
        h = o * np.tanh(c)
        out[t] = h
    return out


def ctc_greedy_decode(logits_tc):
    """logits_tc: (T, NUM_CLASSES). Returns string."""
    idxs = logits_tc.argmax(axis=1)
    chars = []
    prev = -1
    for k in idxs:
        k = int(k)
        if k != prev and k != BLANK:
            chars.append(IDX_TO_CHAR[k])
        prev = k
    return "".join(chars)


# ---------------------------------------------------------------------------
# Full forward
# ---------------------------------------------------------------------------

def forward(x_chw, weights):
    x = np.asarray(x_chw, dtype=np.float32)

    out = resblock(x, weights, "rb1", has_skip_conv=True)
    out = maxpool(out, 2, 2)                            # 32 x 24 x 128

    out = resblock(out, weights, "rb2", has_skip_conv=True)
    out = maxpool(out, 2, 2)                            # 64 x 12 x 64

    out = resblock(out, weights, "rb3", has_skip_conv=True)
    out = maxpool(out, 2, 1)                            # 128 x 6 x 64

    out = resblock(out, weights, "rb4", has_skip_conv=True)
    out = maxpool(out, 2, 1)                            # 256 x 3 x 64

    # AdaptiveAvgPool((1, None)) over H=3 → H=1.
    out = out.mean(axis=1)                              # (256, 64)

    # Sequence: T=64 vectors of C=256.
    seq = out.T                                         # (64, 256)

    fwd = lstm_run(seq, weights["lstm.fwd.weight_ih"], weights["lstm.fwd.weight_hh"],
                   weights["lstm.fwd.bias"], hidden=128)
    bwd = lstm_run(seq[::-1], weights["lstm.bwd.weight_ih"], weights["lstm.bwd.weight_hh"],
                   weights["lstm.bwd.bias"], hidden=128)[::-1]

    rnn_out = np.concatenate([fwd, bwd], axis=1)        # (64, 256)
    logits = rnn_out @ weights["fc.weight"].T + weights["fc.bias"]  # (64, 29)
    return logits


def predict(x_chw, weights):
    return ctc_greedy_decode(forward(x_chw, weights))
