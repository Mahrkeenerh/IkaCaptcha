"""
K-fold cross-validation for label quality checking.

Merges all labeled samples into one pool, runs N training runs with
different random 80/20 splits, and flags samples that are consistently
predicted wrong (likely mislabeled).

Usage: python scripts/kfold_validate.py
"""

import csv
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from PIL import Image
from torchvision import transforms
import torchvision.transforms.v2 as T

from ikaptcha import (
    CRNN, NUM_CLASSES, BLANK, IMG_W, IMG_H,
    char_to_idx, greedy_decode,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_FOLDS = 10
PRETRAIN_EPOCHS = 30
MIXED_EPOCHS = 20
BATCH_SIZE = 32
SYNTHETIC_DIR = "dataset_synthetic/train"
DATASET_DIR = Path("data/dataset_pseudo_v2")


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class AllSamplesDataset(Dataset):
    def __init__(self):
        self.samples = []
        for split in ["train", "val"]:
            img_dir = DATASET_DIR / "images" / split
            lbl_dir = DATASET_DIR / "text_labels" / split
            for img_path in sorted(img_dir.glob("*.png")):
                lbl_path = lbl_dir / (img_path.stem + ".txt")
                if lbl_path.exists():
                    label = lbl_path.read_text().strip().lower()
                    if all(c in char_to_idx for c in label):
                        sample_id = f"{split}/{img_path.name}"
                        self.samples.append((str(img_path), label, sample_id))

    def __len__(self):
        return len(self.samples)


class FilenameDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.samples = []
        for fname in os.listdir(img_dir):
            if not fname.lower().endswith(".png"):
                continue
            parts = fname.rsplit(".", 1)[0].split("_", 1)
            if len(parts) == 2:
                label = parts[1].lower()
                if all(c in char_to_idx for c in label):
                    self.samples.append((fname, label))
        self.samples.sort()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        img = Image.open(os.path.join(self.img_dir, fname)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        target = [char_to_idx[c] for c in label]
        return img, torch.tensor(target, dtype=torch.long), len(target), label, f"synth/{fname}"


class TransformSubset(Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img_path, label, sample_id = self.dataset.samples[self.indices[idx]]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        target = [char_to_idx[c] for c in label]
        return img, torch.tensor(target, dtype=torch.long), len(target), label, sample_id


def collate_fn(batch):
    images, targets, target_lengths, labels, sample_ids = zip(*batch)
    images = torch.stack(images, 0)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    targets = torch.cat(targets, 0)
    return images, targets, target_lengths, labels, sample_ids


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

train_transform = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W)),
    transforms.RandomAffine(degrees=5, translate=(0.03, 0.05), scale=(0.95, 1.05), shear=3),
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.05),
    transforms.ToTensor(),
    transforms.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.3),
    T.GaussianNoise(mean=0.0, sigma=0.03, clip=True),
    transforms.RandomErasing(p=0.15, scale=(0.02, 0.06)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    for images, targets, target_lengths, _, _ in loader:
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)
        target_lengths = target_lengths.to(DEVICE)
        logits = model(images)
        T_, B, _ = logits.shape
        input_lengths = torch.full((B,), T_, dtype=torch.long, device=DEVICE)
        loss = criterion(logits.log_softmax(2), targets, input_lengths, target_lengths)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()


@torch.no_grad()
def predict_all(model, loader):
    model.eval()
    results = []
    for images, _targets, _target_lengths, labels, sample_ids in loader:
        images = images.to(DEVICE)
        logits = model(images)
        preds = greedy_decode(logits)
        for sid, gt, pred in zip(sample_ids, labels, preds):
            results.append((sid, gt, pred))
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading all labeled samples...")
    all_ds = AllSamplesDataset()
    n = len(all_ds)
    print(f"Total: {n} samples")

    synth_ds = FilenameDataset(SYNTHETIC_DIR, transform=train_transform)
    print(f"Synthetic: {len(synth_ds)} samples")

    predictions = defaultdict(list)

    for fold in range(N_FOLDS):
        print(f"\n{'='*60}\nFOLD {fold+1}/{N_FOLDS}\n{'='*60}")
        fold_start = time.time()

        indices = list(range(n))
        random.seed(fold * 42 + 7)
        random.shuffle(indices)
        split = int(0.8 * n)
        train_indices = indices[:split]
        val_indices = indices[split:]

        train_real = TransformSubset(all_ds, train_indices, train_transform)
        val_set = TransformSubset(all_ds, val_indices, val_transform)

        combined = ConcatDataset([synth_ds, train_real])
        weights = [1.0] * len(synth_ds) + [3.0 * len(synth_ds) / len(train_real)] * len(train_real)
        sampler = WeightedRandomSampler(weights, num_samples=len(combined), replacement=True)

        train_loader = DataLoader(combined, batch_size=BATCH_SIZE, sampler=sampler,
                                  num_workers=4, collate_fn=collate_fn, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=4, collate_fn=collate_fn, pin_memory=True)
        pretrain_loader = DataLoader(synth_ds, batch_size=BATCH_SIZE, shuffle=True,
                                     num_workers=4, collate_fn=collate_fn, pin_memory=True)

        model = CRNN(NUM_CLASSES, hidden_size=128).to(DEVICE)
        criterion = nn.CTCLoss(blank=BLANK, zero_infinity=True)

        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, PRETRAIN_EPOCHS, eta_min=1e-6)
        for ep in range(1, PRETRAIN_EPOCHS + 1):
            train_one_epoch(model, pretrain_loader, criterion, optimizer)
            scheduler.step()
            print(f"    pretrain {ep:2d}/{PRETRAIN_EPOCHS}", flush=True)

        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, MIXED_EPOCHS, eta_min=1e-6)
        for ep in range(1, MIXED_EPOCHS + 1):
            train_one_epoch(model, train_loader, criterion, optimizer)
            scheduler.step()
            print(f"    mixed {ep:2d}/{MIXED_EPOCHS}", flush=True)

        results = predict_all(model, val_loader)
        correct = sum(1 for _, gt, pred in results if gt == pred)
        print(f"  Val accuracy: {correct}/{len(results)} = {100*correct/len(results):.1f}%  "
              f"(fold took {time.time()-fold_start:.0f}s)")

        for sid, gt, pred in results:
            predictions[sid].append((fold, pred))

    # Analysis
    print(f"\n{'='*60}\nCROSS-VALIDATION ANALYSIS\n{'='*60}\n")
    id_to_label = {s[2]: s[1] for s in all_ds.samples}
    sample_stats = []
    for sid, label in id_to_label.items():
        preds = predictions.get(sid, [])
        if not preds:
            continue
        n_val = len(preds)
        n_wrong = sum(1 for _, pred in preds if pred != label)
        sample_stats.append({
            "sample_id": sid, "label": label, "n_val": n_val, "n_wrong": n_wrong,
            "error_rate": n_wrong / n_val,
            "wrong_preds": [pred for _, pred in preds if pred != label],
            "most_common_pred": max(set(p for _, p in preds),
                                    key=lambda x: sum(1 for _, p in preds if p == x)),
        })
    sample_stats.sort(key=lambda s: (-s["error_rate"], -s["n_wrong"]))

    always_correct = sum(1 for s in sample_stats if s["n_wrong"] == 0)
    print(f"Total: {len(sample_stats)} | always correct: {always_correct}")

    for header, lo, hi in [("LIKELY MISLABELED (>=80%)", 0.8, 1.01),
                           ("SUSPICIOUS (50-79%)", 0.5, 0.8),
                           ("HARD BUT PROBABLY CORRECT (30-49%)", 0.3, 0.5)]:
        print(f"\n{header}:")
        print("-" * 90)
        for s in sample_stats:
            if lo <= s["error_rate"] < hi:
                print(f"{s['sample_id']:40s} {s['label']:12s} {s['most_common_pred']:12s} "
                      f"{s['n_wrong']:3d}/{s['n_val']:3d}  {s['error_rate']:6.1%}")

    csv_path = "kfold_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "label", "most_common_pred",
                                                "n_wrong", "n_val", "error_rate", "wrong_preds"])
        writer.writeheader()
        for s in sample_stats:
            row = dict(s)
            row["wrong_preds"] = "|".join(row["wrong_preds"])
            writer.writerow(row)
    print(f"\nFull results saved to {csv_path}")


if __name__ == "__main__":
    main()
