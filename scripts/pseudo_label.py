"""
Run inference on raw real captchas and save predicted labels + per-sequence
confidence to a CSV. Used to bootstrap pseudo-labeled training data.

Confidence is the mean softmax probability at each non-blank CTC timestep
along the greedy-decoded path.

Usage:
    python scripts/pseudo_label.py [--checkpoint models/ikaptcha.pth] [--batch-size 64]
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from PIL import Image

from ikaptcha import (
    CRNN, NUM_CLASSES, BLANK, idx_to_char, val_transform,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def greedy_decode_with_confidence(logits):
    """Greedy CTC decode + per-sequence mean confidence."""
    probs = F.softmax(logits, dim=2)         # (T, B, C)
    indices = logits.argmax(dim=2)            # (T, B)
    probs_t = probs.permute(1, 0, 2)         # (B, T, C)
    indices_t = indices.permute(1, 0)        # (B, T)

    results = []
    for b in range(indices_t.shape[0]):
        seq = indices_t[b].tolist()
        prob_seq = probs_t[b]
        chars = []
        char_confidences = []
        prev = None
        for t, idx in enumerate(seq):
            if idx != prev and idx != BLANK:
                chars.append(idx_to_char[idx])
                char_confidences.append(prob_seq[t, idx].item())
            prev = idx
        label = "".join(chars)
        confidence = sum(char_confidences) / len(char_confidences) if char_confidences else 0.0
        results.append((label, confidence))
    return results


def collect_png_paths(*dirs):
    paths = []
    for d in dirs:
        d = Path(d)
        if not d.exists():
            print(f"  Warning: directory not found, skipping: {d}")
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() == ".png":
                paths.append(str(p.resolve()))
    return paths


def load_batch(paths):
    tensors = []
    valid = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            tensors.append(val_transform(img))
            valid.append(p)
        except Exception as e:
            print(f"  Warning: could not load {p}: {e}")
    if not tensors:
        return None, []
    return torch.stack(tensors, 0), valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/ikaptcha.pth")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", default="pseudo_labels.csv")
    parser.add_argument("--input-dirs", nargs="+",
                        default=["data/real_samples_unlabeled", "data/real_samples"],
                        help="Directories of raw PNGs to label")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {args.checkpoint}")

    model = CRNN(NUM_CLASSES, hidden_size=128).to(DEVICE)
    state = torch.load(args.checkpoint, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")

    all_paths = collect_png_paths(*args.input_dirs)
    print(f"\nImages found: {len(all_paths)}")

    rows = []
    total = len(all_paths)
    bs = args.batch_size

    print(f"\nRunning inference (batch_size={bs})...")
    for start in range(0, total, bs):
        batch_paths = all_paths[start:start + bs]
        images, valid_paths = load_batch(batch_paths)
        if images is None:
            continue
        images = images.to(DEVICE)
        with torch.no_grad():
            logits = model(images)
        decoded = greedy_decode_with_confidence(logits)
        for path, (label, confidence) in zip(valid_paths, decoded):
            rows.append((path, label, confidence))
        if (start // bs) % 10 == 0:
            done = min(start + bs, total)
            print(f"  {done}/{total} ({100*done/total:.0f}%)")

    print(f"  Done — {len(rows)} images processed")

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "predicted_label", "confidence"])
        writer.writerows(rows)
    print(f"\nSaved: {args.output}")

    confidences = [r[2] for r in rows]
    if confidences:
        print("\nConfidence distribution:")
        print(f"  Mean:   {sum(confidences)/len(confidences):.4f}")
        print(f"  Median: {sorted(confidences)[len(confidences)//2]:.4f}")
        print(f"  Min:    {min(confidences):.4f}")
        print(f"  Max:    {max(confidences):.4f}\n")
        print("  Threshold | Pass | Pass%")
        print("  ----------+------+------")
        for t in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]:
            passing = sum(1 for c in confidences if c >= t)
            print(f"  >= {t:.2f}   | {passing:4d} | {100*passing/len(confidences):.1f}%")


if __name__ == "__main__":
    main()
