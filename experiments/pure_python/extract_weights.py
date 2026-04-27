"""
One-shot extraction: PyTorch CRNN .pth → folded weights.bin + manifest.

Folds BatchNorm into the preceding Conv2d for inference. Writes a flat
float32 little-endian blob and a JSON manifest mapping tensor name →
(byte_offset, shape).

Usage:
    python experiments/pure_python/extract_weights.py
"""

import json
import math
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import torch

from ikaptcha import CRNN, NUM_CLASSES


def fold_bn_into_conv(conv_w, conv_b, bn_w, bn_b, bn_mean, bn_var, eps):
    """
    Fold a Conv2d (with bias) followed by a BatchNorm2d into a single Conv2d.
        y = bn(conv(x))
        let scale = bn_w / sqrt(bn_var + eps)
        new_w[c] = conv_w[c] * scale[c]
        new_b[c] = (conv_b[c] - bn_mean[c]) * scale[c] + bn_b[c]
    """
    scale = bn_w / torch.sqrt(bn_var + eps)
    new_w = conv_w * scale.view(-1, 1, 1, 1)
    new_b = (conv_b - bn_mean) * scale + bn_b
    return new_w, new_b


def main():
    ckpt = ROOT / "models" / "ikaptcha.pth"
    out_bin = Path(__file__).parent / "weights.bin"
    out_manifest = Path(__file__).parent / "weights_manifest.json"

    model = CRNN(NUM_CLASSES, hidden_size=128)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    tensors = []  # list of (name, flat_float_list, shape_tuple)

    def add(name, t):
        flat = t.detach().cpu().contiguous().view(-1).tolist()
        shape = tuple(t.shape)
        tensors.append((name, flat, shape))

    # CNN: 4 ResBlocks. Indices in the Sequential: 0, 2, 5, 8 are the ResBlocks.
    rb_indices = [0, 2, 5, 8]
    rb_names = ["rb1", "rb2", "rb3", "rb4"]

    for idx, name in zip(rb_indices, rb_names):
        rb = model.cnn[idx]
        eps1 = rb.bn1.eps
        eps2 = rb.bn2.eps

        w1, b1 = fold_bn_into_conv(
            rb.conv1.weight.data, rb.conv1.bias.data,
            rb.bn1.weight.data, rb.bn1.bias.data,
            rb.bn1.running_mean.data, rb.bn1.running_var.data, eps1,
        )
        w2, b2 = fold_bn_into_conv(
            rb.conv2.weight.data, rb.conv2.bias.data,
            rb.bn2.weight.data, rb.bn2.bias.data,
            rb.bn2.running_mean.data, rb.bn2.running_var.data, eps2,
        )
        add(f"{name}.conv1.weight", w1)
        add(f"{name}.conv1.bias", b1)
        add(f"{name}.conv2.weight", w2)
        add(f"{name}.conv2.bias", b2)

        # Skip: Conv2d(1x1) when channel mismatch, else Identity.
        if isinstance(rb.skip, torch.nn.Conv2d):
            add(f"{name}.skip.weight", rb.skip.weight.data)
            add(f"{name}.skip.bias", rb.skip.bias.data)

        # SE block.
        add(f"{name}.se.fc1.weight", rb.se.fc1.weight.data)
        add(f"{name}.se.fc1.bias", rb.se.fc1.bias.data)
        add(f"{name}.se.fc2.weight", rb.se.fc2.weight.data)
        add(f"{name}.se.fc2.bias", rb.se.fc2.bias.data)

    # LSTM: 1-layer bidirectional. Pre-sum bias_ih and bias_hh per direction.
    lstm = model.rnn
    add("lstm.fwd.weight_ih", lstm.weight_ih_l0.data)
    add("lstm.fwd.weight_hh", lstm.weight_hh_l0.data)
    add("lstm.fwd.bias", lstm.bias_ih_l0.data + lstm.bias_hh_l0.data)
    add("lstm.bwd.weight_ih", lstm.weight_ih_l0_reverse.data)
    add("lstm.bwd.weight_hh", lstm.weight_hh_l0_reverse.data)
    add("lstm.bwd.bias", lstm.bias_ih_l0_reverse.data + lstm.bias_hh_l0_reverse.data)

    # FC head.
    add("fc.weight", model.fc.weight.data)
    add("fc.bias", model.fc.bias.data)

    # Pack to binary, build manifest.
    manifest = []
    offset = 0
    blob = bytearray()
    for name, flat, shape in tensors:
        n = len(flat)
        manifest.append({"name": name, "offset": offset, "shape": list(shape), "n": n})
        blob.extend(struct.pack(f"<{n}f", *flat))
        offset += n * 4

    out_bin.write_bytes(blob)
    out_manifest.write_text(json.dumps(manifest, indent=2))

    total_floats = offset // 4
    print(f"Wrote {out_bin} ({offset / 1024:.1f} KB, {total_floats:,} float32s)")
    print(f"Wrote {out_manifest} ({len(manifest)} tensors)")

    # Sanity vs nominal param count.
    nominal = sum(p.numel() for p in model.parameters())
    print(f"Model nominal params: {nominal:,}")
    print(f"Extracted floats:     {total_floats:,}  "
          f"(diff is BN running_mean/var folded into conv)")


if __name__ == "__main__":
    main()
