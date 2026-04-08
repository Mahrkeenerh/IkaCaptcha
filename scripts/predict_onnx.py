"""
ONNX single-image inference (production / lightweight path).

Depends only on onnxruntime, Pillow, and numpy. No PyTorch needed. Use this
as the reference implementation when porting to other runtimes (browser via
onnxruntime-web, mobile, etc).

Usage:
    python scripts/predict_onnx.py data/samples/test1.png data/samples/test2.png
    python scripts/predict_onnx.py --model models/crnn.onnx data/samples/*.png
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import onnxruntime as ort
from PIL import Image

from ikaptcha import greedy_decode_numpy, preprocess_pil


def confidence(logits_tbc: np.ndarray) -> float:
    """Mean of per-timestep max softmax — single-image only."""
    x = logits_tbc[:, 0, :]
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    p = e / e.sum(axis=1, keepdims=True)
    return float(p.max(axis=1).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/crnn.onnx")
    parser.add_argument("images", nargs="+")
    args = parser.parse_args()

    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])

    for path in args.images:
        img = Image.open(path)
        x = preprocess_pil(img)
        logits = session.run(["logits"], {"input": x})[0]  # (T, 1, C)
        pred = greedy_decode_numpy(logits)[0]
        conf = confidence(logits)
        print(f"{path}: {pred}  (conf={conf:.3f})")


if __name__ == "__main__":
    main()
