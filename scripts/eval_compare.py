"""
Evaluate a checkpoint on both the original YOLO val set and our corrected
val set.

Usage:
    python scripts/eval_compare.py [--checkpoint models/final_mixed.pth]
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image

from ikaptcha import (
    CRNN, NUM_CLASSES, char_to_idx, greedy_decode, val_transform,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_val(img_dir, lbl_dir):
    samples = []
    for img_path in sorted(Path(img_dir).glob("*.png")):
        lbl_path = Path(lbl_dir) / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        label = lbl_path.read_text().strip().lower()
        if all(c in char_to_idx for c in label):
            samples.append((str(img_path), label, img_path.stem))
    return samples


@torch.no_grad()
def evaluate(model, samples):
    model.eval()
    correct = 0
    errors = []
    for img_path, label, name in samples:
        img = Image.open(img_path).convert("RGB")
        img_t = val_transform(img).unsqueeze(0).to(DEVICE)
        logits = model(img_t)
        pred = greedy_decode(logits)[0]
        if pred == label:
            correct += 1
        else:
            errors.append((name, label, pred))
    return correct, len(samples), errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/final_mixed.pth")
    args = parser.parse_args()

    original = load_val(
        "data/ikariam_pirate_captcha_dataset/images/val",
        "data/ikariam_pirate_captcha_dataset/text_labels/val",
    )
    corrected = load_val(
        "data/dataset_pseudo_v2/images/val",
        "data/dataset_pseudo_v2/text_labels/val",
    )
    print(f"Original YOLO val:  {len(original)} samples")
    print(f"Corrected val:      {len(corrected)} samples")

    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        return

    model = CRNN(NUM_CLASSES, hidden_size=128).to(DEVICE)
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE, weights_only=True))
    print(f"\nLoaded: {args.checkpoint}\n")

    c, t, errs = evaluate(model, original)
    print(f"Original val:  {c}/{t} = {100*c/t:.1f}%")

    c2, t2, errs2 = evaluate(model, corrected)
    print(f"Corrected val: {c2}/{t2} = {100*c2/t2:.1f}%")

    if errs2:
        print(f"\nErrors on corrected val ({len(errs2)}):")
        for n, gt, pred in errs2:
            print(f"  {n}: label={gt} pred={pred}")


if __name__ == "__main__":
    main()
