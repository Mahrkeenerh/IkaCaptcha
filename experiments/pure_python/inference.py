"""
Pure-Python CRNN inference. Stdlib only (struct, math, json, array).

Layout convention: every tensor is a flat Python list of floats.
Activations are stored as (C, H, W) with the linear index c*H*W + h*W + w.
Conv weights are (C_out, C_in, kH, kW) with index oc*Cin*kHW + ic*kHW + kh*kW + kw.

Public entry point:
    predict(image_chw_flat, weights) -> str
where image_chw_flat is the preprocessed input (3*48*256 floats, in [-1, 1]).
"""

import json
import math
import struct
from pathlib import Path

CHARSET = "abcdefghjklmnpqrstuvwxy23457"
BLANK = 0
NUM_CLASSES = len(CHARSET) + 1  # 29
IDX_TO_CHAR = {i + 1: c for i, c in enumerate(CHARSET)}

IMG_H = 48
IMG_W = 256


# ---------------------------------------------------------------------------
# Weight loader
# ---------------------------------------------------------------------------

def load_weights(blob_path: str | Path = None, manifest_path: str | Path = None) -> dict:
    here = Path(__file__).parent
    blob_path = Path(blob_path) if blob_path else here / "weights.bin"
    manifest_path = Path(manifest_path) if manifest_path else here / "weights_manifest.json"

    blob = blob_path.read_bytes()
    manifest = json.loads(manifest_path.read_text())

    out = {}
    for entry in manifest:
        n = entry["n"]
        offset = entry["offset"]
        flat = list(struct.unpack_from(f"<{n}f", blob, offset))
        out[entry["name"]] = (flat, tuple(entry["shape"]))
    return out


# ---------------------------------------------------------------------------
# Pointwise / pooling ops
# ---------------------------------------------------------------------------

def relu_inplace(x):
    for i in range(len(x)):
        if x[i] < 0.0:
            x[i] = 0.0


def sigmoid_list(x):
    return [1.0 / (1.0 + math.exp(-v)) if v >= 0 else math.exp(v) / (1.0 + math.exp(v)) for v in x]


def maxpool(x, C, H, W, kh, kw):
    """Stride = kernel. Returns (C, H//kh, W//kw) flat."""
    H2 = H // kh
    W2 = W // kw
    out = [0.0] * (C * H2 * W2)
    for c in range(C):
        in_base = c * H * W
        out_base = c * H2 * W2
        for i in range(H2):
            ih_start = i * kh
            for j in range(W2):
                iw_start = j * kw
                m = -float("inf")
                for dh in range(kh):
                    row = in_base + (ih_start + dh) * W
                    for dw in range(kw):
                        v = x[row + iw_start + dw]
                        if v > m:
                            m = v
                out[out_base + i * W2 + j] = m
    return out


def avgpool_global_chw(x, C, H, W):
    """Collapse to (C,) by averaging H*W per channel."""
    out = [0.0] * C
    inv = 1.0 / (H * W)
    for c in range(C):
        s = 0.0
        base = c * H * W
        for k in range(H * W):
            s += x[base + k]
        out[c] = s * inv
    return out


def avgpool_h_to_1(x, C, H, W):
    """Collapse (C, H, W) → (C, 1, W) by averaging over H."""
    out = [0.0] * (C * W)
    inv = 1.0 / H
    for c in range(C):
        in_base = c * H * W
        out_base = c * W
        for j in range(W):
            s = 0.0
            for i in range(H):
                s += x[in_base + i * W + j]
            out[out_base + j] = s * inv
    return out


# ---------------------------------------------------------------------------
# Convolutions
# ---------------------------------------------------------------------------

def conv2d_3x3_pad1(x, w, b, C_in, C_out, H, W):
    """3x3, stride 1, padding 1. Output shape == input H, W."""
    HW = H * W
    Cin9 = C_in * 9
    out = [0.0] * (C_out * HW)

    for oc in range(C_out):
        bias = b[oc]
        out_base = oc * HW
        w_oc = oc * Cin9
        for h in range(H):
            h_top = h - 1
            h_bot = h + 1
            for ww in range(W):
                acc = bias
                w_left = ww - 1
                w_right = ww + 1
                for ic in range(C_in):
                    in_base = ic * HW
                    w_ic = w_oc + ic * 9
                    # kh=0
                    if h_top >= 0:
                        row = in_base + h_top * W
                        if w_left >= 0:
                            acc += x[row + w_left] * w[w_ic]
                        acc += x[row + ww] * w[w_ic + 1]
                        if w_right < W:
                            acc += x[row + w_right] * w[w_ic + 2]
                    # kh=1
                    row = in_base + h * W
                    if w_left >= 0:
                        acc += x[row + w_left] * w[w_ic + 3]
                    acc += x[row + ww] * w[w_ic + 4]
                    if w_right < W:
                        acc += x[row + w_right] * w[w_ic + 5]
                    # kh=2
                    if h_bot < H:
                        row = in_base + h_bot * W
                        if w_left >= 0:
                            acc += x[row + w_left] * w[w_ic + 6]
                        acc += x[row + ww] * w[w_ic + 7]
                        if w_right < W:
                            acc += x[row + w_right] * w[w_ic + 8]
                out[out_base + h * W + ww] = acc
    return out


def conv2d_1x1(x, w, b, C_in, C_out, H, W):
    """1x1 conv. Equivalent to per-pixel linear over channels."""
    HW = H * W
    out = [0.0] * (C_out * HW)
    for oc in range(C_out):
        bias = b[oc]
        out_base = oc * HW
        w_oc = oc * C_in
        for p in range(HW):
            acc = bias
            for ic in range(C_in):
                acc += x[ic * HW + p] * w[w_oc + ic]
            out[out_base + p] = acc
    return out


# ---------------------------------------------------------------------------
# Linear / FC
# ---------------------------------------------------------------------------

def linear(x, w, b, in_f, out_f):
    """y = w @ x + b. w shape (out, in) flat, x len in, b len out."""
    out = list(b)  # copy
    for o in range(out_f):
        s = 0.0
        w_base = o * in_f
        for i in range(in_f):
            s += w[w_base + i] * x[i]
        out[o] += s
    return out


# ---------------------------------------------------------------------------
# SE block + ResBlock
# ---------------------------------------------------------------------------

def se_block(x, weights, prefix, C, H, W):
    """Squeeze-Excitation: channel attention. Modifies x in place by mult."""
    fc1_w, _ = weights[f"{prefix}.se.fc1.weight"]
    fc1_b, _ = weights[f"{prefix}.se.fc1.bias"]
    fc2_w, _ = weights[f"{prefix}.se.fc2.weight"]
    fc2_b, _ = weights[f"{prefix}.se.fc2.bias"]
    mid = len(fc1_b)

    pooled = avgpool_global_chw(x, C, H, W)
    h1 = linear(pooled, fc1_w, fc1_b, C, mid)
    relu_inplace(h1)
    h2 = linear(h1, fc2_w, fc2_b, mid, C)
    scale = sigmoid_list(h2)

    HW = H * W
    for c in range(C):
        s = scale[c]
        base = c * HW
        for k in range(HW):
            x[base + k] *= s
    return x


def resblock(x, weights, prefix, C_in, C_out, H, W, has_skip_conv):
    """Two 3x3 convs (BN folded), SE, residual, ReLU. Returns out flat list."""
    cw1, _ = weights[f"{prefix}.conv1.weight"]
    cb1, _ = weights[f"{prefix}.conv1.bias"]
    cw2, _ = weights[f"{prefix}.conv2.weight"]
    cb2, _ = weights[f"{prefix}.conv2.bias"]

    # Residual path
    if has_skip_conv:
        sw, _ = weights[f"{prefix}.skip.weight"]
        sb, _ = weights[f"{prefix}.skip.bias"]
        residual = conv2d_1x1(x, sw, sb, C_in, C_out, H, W)
    else:
        residual = list(x)  # identity, same shape, already C_out

    # Main path
    out = conv2d_3x3_pad1(x, cw1, cb1, C_in, C_out, H, W)
    relu_inplace(out)
    out = conv2d_3x3_pad1(out, cw2, cb2, C_out, C_out, H, W)
    se_block(out, weights, prefix, C_out, H, W)

    # Add residual + ReLU
    for i in range(len(out)):
        v = out[i] + residual[i]
        out[i] = v if v > 0.0 else 0.0
    return out


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------

def lstm_run(seq, w_ih, w_hh, b, hidden):
    """
    seq: list of T vectors, each of length input_size.
    w_ih: flat (4*hidden * input_size). w_hh: flat (4*hidden * hidden).
    b: flat (4*hidden,).
    Returns list of T vectors, each of length hidden.
    """
    if not seq:
        return []
    input_size = len(seq[0])
    h = [0.0] * hidden
    c = [0.0] * hidden
    out = []

    H4 = 4 * hidden
    h0 = 0
    h1 = hidden
    h2 = 2 * hidden
    h3 = 3 * hidden

    for x_t in seq:
        # gates = w_ih @ x + w_hh @ h + b   (length 4*hidden)
        gates = list(b)
        for r in range(H4):
            ih_base = r * input_size
            hh_base = r * hidden
            s = 0.0
            for k in range(input_size):
                s += w_ih[ih_base + k] * x_t[k]
            for k in range(hidden):
                s += w_hh[hh_base + k] * h[k]
            gates[r] += s

        new_h = [0.0] * hidden
        new_c = [0.0] * hidden
        for k in range(hidden):
            i = 1.0 / (1.0 + math.exp(-gates[h0 + k]))
            f = 1.0 / (1.0 + math.exp(-gates[h1 + k]))
            g = math.tanh(gates[h2 + k])
            o = 1.0 / (1.0 + math.exp(-gates[h3 + k]))
            new_c[k] = f * c[k] + i * g
            new_h[k] = o * math.tanh(new_c[k])
        h, c = new_h, new_c
        out.append(h)
    return out


# ---------------------------------------------------------------------------
# CTC decode
# ---------------------------------------------------------------------------

def ctc_greedy_decode(logits_tc):
    """logits_tc: list of T vectors of length NUM_CLASSES. Returns string."""
    chars = []
    prev = -1
    for vec in logits_tc:
        # argmax
        best = 0
        best_v = vec[0]
        for k in range(1, len(vec)):
            if vec[k] > best_v:
                best_v = vec[k]
                best = k
        if best != prev and best != BLANK:
            chars.append(IDX_TO_CHAR[best])
        prev = best
    return "".join(chars)


# ---------------------------------------------------------------------------
# Full forward
# ---------------------------------------------------------------------------

def forward(x_chw, weights):
    """x_chw: flat list of 3*48*256 floats, normalized to [-1, 1].
    Returns list of 64 vectors of 29 logits each (T, C)."""

    # CNN block 1: 3 → 32, no skip conv mismatch? in_ch=3, out_ch=32, so yes skip conv.
    out = resblock(x_chw, weights, "rb1", 3, 32, 48, 256, has_skip_conv=True)
    out = maxpool(out, 32, 48, 256, 2, 2)            # → 32 × 24 × 128

    out = resblock(out, weights, "rb2", 32, 64, 24, 128, has_skip_conv=True)
    out = maxpool(out, 64, 24, 128, 2, 2)            # → 64 × 12 × 64

    out = resblock(out, weights, "rb3", 64, 128, 12, 64, has_skip_conv=True)
    out = maxpool(out, 128, 12, 64, 2, 1)            # → 128 × 6 × 64

    out = resblock(out, weights, "rb4", 128, 256, 6, 64, has_skip_conv=True)
    out = maxpool(out, 256, 6, 64, 2, 1)             # → 256 × 3 × 64

    # AdaptiveAvgPool((1, None)) over H=3 → H=1.
    out = avgpool_h_to_1(out, 256, 3, 64)            # → 256 × 64 (flat)

    # Reshape to sequence: T=64 vectors of C=256.
    # Memory layout is c*64 + t, but we want list of T vectors of [c0..c255].
    seq = []
    for t in range(64):
        seq.append([out[c * 64 + t] for c in range(256)])

    # BiLSTM
    fwd_w_ih, _ = weights["lstm.fwd.weight_ih"]
    fwd_w_hh, _ = weights["lstm.fwd.weight_hh"]
    fwd_b, _ = weights["lstm.fwd.bias"]
    bwd_w_ih, _ = weights["lstm.bwd.weight_ih"]
    bwd_w_hh, _ = weights["lstm.bwd.weight_hh"]
    bwd_b, _ = weights["lstm.bwd.bias"]

    fwd_out = lstm_run(seq, fwd_w_ih, fwd_w_hh, fwd_b, hidden=128)
    bwd_out = lstm_run(seq[::-1], bwd_w_ih, bwd_w_hh, bwd_b, hidden=128)
    bwd_out.reverse()

    # Concat per timestep → 256-d vectors.
    rnn_out = [fwd_out[t] + bwd_out[t] for t in range(64)]

    # FC head: 256 → 29
    fc_w, _ = weights["fc.weight"]
    fc_b, _ = weights["fc.bias"]
    logits = [linear(v, fc_w, fc_b, 256, NUM_CLASSES) for v in rnn_out]
    return logits


def predict(x_chw, weights):
    return ctc_greedy_decode(forward(x_chw, weights))
