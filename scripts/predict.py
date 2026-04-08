"""
PyTorch single-image inference (development/debugging).

For deployment, prefer scripts/predict_onnx.py — it has no torch dependency
and matches the production ONNX file we ship.

Usage:
    python scripts/predict.py data/samples/test1.png data/samples/test2.png
    python scripts/predict.py --model models/ikaptcha.pth data/samples/*.png
"""

import argparse
import sys
from pathlib import Path

# Bootstrap repo root onto sys.path so `from ikaptcha import ...` works
# without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image

from ikaptcha import CRNN, NUM_CLASSES, greedy_decode, val_transform

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict(model, image_path: str) -> tuple[str, float]:
    img = Image.open(image_path).convert("RGB")
    tensor = val_transform(img).unsqueeze(0).to(DEVICE)
    logits = model(tensor)  # (T, 1, C)
    pred = greedy_decode(logits)[0]
    probs = logits.squeeze(1).softmax(dim=1)
    conf = probs.max(dim=1).values.mean().item()
    return pred, conf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/ikaptcha.pth")
    parser.add_argument("images", nargs="+")
    args = parser.parse_args()

    model = CRNN(NUM_CLASSES, hidden_size=128).to(DEVICE)
    model.load_state_dict(torch.load(args.model, map_location=DEVICE, weights_only=True))
    model.eval()

    for path in args.images:
        label, conf = predict(model, path)
        print(f"{path}: {label}  (conf={conf:.3f})")


if __name__ == "__main__":
    main()
