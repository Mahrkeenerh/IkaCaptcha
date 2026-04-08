"""
Two-phase CRNN training for the Ikariam captcha.

Phase 1: Pretrain on synthetic data (~65k samples).
Phase 2: Mixed real + synthetic with 3× weighted oversampling of real,
         OneCycleLR (no warm restarts, no SWA).

Usage:
    python scripts/train.py --phase all
    python scripts/train.py --phase pretrain
    python scripts/train.py --phase mixed --checkpoint models/best_pretrain.pth
"""

import argparse
import math
import os
import sys
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

# Phase configs: (epochs, lr, weight_decay, batch_size, warmup_epochs)
PHASE_CONFIG = {
    "pretrain": (50, 3e-4, 1e-2, 32, 3),
    "mixed":    (40, 2e-4, 1e-2, 32, 3),
}

# Dataset paths (override real-data root with DATASET_DIR env var if needed)
COMBINED_DIR = os.environ.get("DATASET_DIR", "data/dataset_pseudo_v2")
SYNTHETIC_TRAIN = "dataset_synthetic/train"
SYNTHETIC_VAL = "dataset_synthetic/val"
REAL_TRAIN_IMG = os.path.join(COMBINED_DIR, "images/train")
REAL_TRAIN_LBL = os.path.join(COMBINED_DIR, "text_labels/train")
REAL_VAL_IMG = os.path.join(COMBINED_DIR, "images/val")
REAL_VAL_LBL = os.path.join(COMBINED_DIR, "text_labels/val")

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Eval helper (training-only)
# ---------------------------------------------------------------------------

def edit_distance(a, b):
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class TextLabelDataset(Dataset):
    """Load images with separate text label files (YOLO-converted format)."""
    def __init__(self, img_dir, lbl_dir, transform=None):
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.transform = transform
        self.samples = []
        for fname in os.listdir(img_dir):
            if not fname.lower().endswith(".png"):
                continue
            name = fname.rsplit(".", 1)[0]
            lbl_path = os.path.join(lbl_dir, name + ".txt")
            if os.path.exists(lbl_path):
                with open(lbl_path) as f:
                    label = f.read().strip().lower()
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
        return img, torch.tensor(target, dtype=torch.long), len(target), label


class FilenameDataset(Dataset):
    """Load images where label is encoded in filename: {index}_{label}.png."""
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.samples = []
        for fname in os.listdir(img_dir):
            if not fname.lower().endswith(".png"):
                continue
            name = fname.rsplit(".", 1)[0]
            parts = name.split("_", 1)
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
        return img, torch.tensor(target, dtype=torch.long), len(target), label


def collate_fn(batch):
    images, targets, target_lengths, labels = zip(*batch)
    images = torch.stack(images, 0)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    targets = torch.cat(targets, 0)
    return images, targets, target_lengths, labels


# ---------------------------------------------------------------------------
# Train transforms (val transform lives in ikaptcha.transforms)
# ---------------------------------------------------------------------------

def get_train_transform():
    return transforms.Compose([
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


def get_val_transform():
    return transforms.Compose([
        transforms.Resize((IMG_H, IMG_W)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


# ---------------------------------------------------------------------------
# Training / Validation
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, scheduler_per_batch=None):
    model.train()
    total_loss = 0.0
    count = 0
    for images, targets, target_lengths, _labels in loader:
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)
        target_lengths = target_lengths.to(DEVICE)

        logits = model(images)
        T_, B, _ = logits.shape
        input_lengths = torch.full((B,), T_, dtype=torch.long, device=DEVICE)
        log_probs = logits.log_softmax(2)

        loss = criterion(log_probs, targets, input_lengths, target_lengths)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        if scheduler_per_batch is not None:
            scheduler_per_batch.step()

        total_loss += loss.item() * B
        count += B

    return total_loss / count


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_edit_dist = 0
    total_gt_chars = 0
    total_samples = 0

    for images, targets, target_lengths, labels in loader:
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)
        target_lengths = target_lengths.to(DEVICE)

        logits = model(images)
        T_, B, _ = logits.shape
        input_lengths = torch.full((B,), T_, dtype=torch.long, device=DEVICE)
        log_probs = logits.log_softmax(2)

        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        total_loss += loss.item() * B

        preds = greedy_decode(logits)
        for pred, gt in zip(preds, labels):
            total_samples += 1
            if pred == gt:
                total_correct += 1
            total_edit_dist += edit_distance(pred, gt)
            total_gt_chars += len(gt)

    avg_loss = total_loss / len(loader.dataset)
    seq_acc = total_correct / total_samples
    cer = total_edit_dist / max(1, total_gt_chars)
    return avg_loss, seq_acc, cer


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------

def run_pretrain(model, config, train_tf, val_tf):
    epochs, lr, wd, batch_size, warmup = config
    train_ds = FilenameDataset(SYNTHETIC_TRAIN, transform=train_tf)
    val_ds = TextLabelDataset(REAL_VAL_IMG, REAL_VAL_LBL, transform=val_tf)
    print(f"Pretrain: {len(train_ds)} synthetic train | {len(val_ds)} real val")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, collate_fn=collate_fn, pin_memory=True)

    criterion = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, epochs - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    ckpt_dir = MODELS_DIR / "checkpoints_pretrain"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_seq_acc = 0.0

    print(f"\n{'='*60}")
    print(f"Phase: pretrain | {epochs} epochs | LR={lr} | WD={wd} | BS={batch_size}")
    print(f"{'='*60}\n")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, seq_acc, cer = validate(model, val_loader, criterion)
        lr_now = optimizer.param_groups[0]["lr"]
        scheduler.step()

        tag = ""
        if seq_acc > best_seq_acc:
            best_seq_acc = seq_acc
            torch.save(model.state_dict(), MODELS_DIR / "best_pretrain.pth")
            tag = " *best*"

        if epoch % 10 == 0 or epoch == epochs:
            torch.save(model.state_dict(), ckpt_dir / f"pretrain_ep{epoch:03d}.pth")

        print(f"  [pretrain] {epoch:02d}/{epochs}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  "
              f"seq_acc={seq_acc:.1%}  cer={cer:.4f}  lr={lr_now:.1e}{tag}")

    print(f"\n  Best pretrain accuracy: {best_seq_acc:.1%}")
    return best_seq_acc


def run_mixed(model, config, train_tf, val_tf):
    epochs, lr, wd, batch_size, warmup = config

    real_ds = TextLabelDataset(REAL_TRAIN_IMG, REAL_TRAIN_LBL, transform=train_tf)
    synth_ds = FilenameDataset(SYNTHETIC_TRAIN, transform=train_tf)
    val_ds = TextLabelDataset(REAL_VAL_IMG, REAL_VAL_LBL, transform=val_tf)

    combined_ds = ConcatDataset([synth_ds, real_ds])
    weights = [1.0] * len(synth_ds) + [3.0 * len(synth_ds) / len(real_ds)] * len(real_ds)
    sampler = WeightedRandomSampler(weights, num_samples=len(combined_ds), replacement=True)

    print(f"Mixed: {len(real_ds)} real + {len(synth_ds)} synthetic | {len(val_ds)} val")
    print(f"  Weighted sampling: real 3x oversampled")

    train_loader = DataLoader(combined_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=4, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, collate_fn=collate_fn, pin_memory=True)

    criterion = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        div_factor=10.0,
        final_div_factor=100.0,
    )

    ckpt_dir = MODELS_DIR / "checkpoints_mixed"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_seq_acc = 0.0

    print(f"\n{'='*60}")
    print(f"Phase: mixed | {epochs} epochs | OneCycleLR max_lr={lr} | WD={wd} | BS={batch_size}")
    print(f"  steps/epoch={steps_per_epoch}, pct_start=0.1 (warmup 10%)")
    print(f"{'='*60}\n")

    seq_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer,
                                     scheduler_per_batch=scheduler)
        val_loss, seq_acc, cer = validate(model, val_loader, criterion)
        lr_now = optimizer.param_groups[0]["lr"]

        tag = ""
        if seq_acc > best_seq_acc:
            best_seq_acc = seq_acc
            torch.save(model.state_dict(), MODELS_DIR / "best_mixed.pth")
            tag = " *best*"

        if epoch % 10 == 0 or epoch == epochs:
            torch.save(model.state_dict(), ckpt_dir / f"mixed_ep{epoch:03d}.pth")

        print(f"  [mixed] {epoch:02d}/{epochs}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  "
              f"seq_acc={seq_acc:.1%}  cer={cer:.4f}  lr={lr_now:.1e}{tag}")

    torch.save(model.state_dict(), MODELS_DIR / "final_mixed.pth")
    print(f"\n  Best mixed accuracy: {best_seq_acc:.1%}")
    print(f"  Final-epoch accuracy: {seq_acc:.1%}  (saved as models/final_mixed.pth)")
    return best_seq_acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Two-phase CRNN training")
    parser.add_argument("--phase", choices=["pretrain", "mixed", "all"], default="all")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint to load before starting")
    args = parser.parse_args()

    train_tf = get_train_transform()
    val_tf = get_val_transform()

    model = CRNN(NUM_CLASSES, hidden_size=128).to(DEVICE)
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE, weights_only=True))

    phases = ["pretrain", "mixed"] if args.phase == "all" else [args.phase]

    for phase in phases:
        config = PHASE_CONFIG[phase]
        if phase == "pretrain":
            run_pretrain(model, config, train_tf, val_tf)
            if "mixed" in phases:
                model.load_state_dict(torch.load(MODELS_DIR / "best_pretrain.pth",
                                      map_location=DEVICE, weights_only=True))
        elif phase == "mixed":
            run_mixed(model, config, train_tf, val_tf)

    print("\nDone! Production model saved as models/final_mixed.pth")


if __name__ == "__main__":
    main()
