"""
Assemble training dataset v2 with confidence-filtered pseudo-labels.

Train: YOLO original train (1200) + reviewed (99) + pseudo-labeled at >=0.99 confidence
Val:   Original YOLO val (300, uncorrected)

Output: data/dataset_pseudo_v2/

Note: this is a one-shot historical builder. The output dataset is shipped
with the repo, so you only need to re-run this if you fetch + pseudo-label
fresh data and want to rebuild from scratch. It expects the intermediate
artifacts (`data/real_samples_reviewed/`, `pseudo_labels.csv`) which are not
committed — you have to regenerate them via fetch_captchas.py + pseudo_label.py.
"""

import csv
import shutil
from pathlib import Path

OUT = Path("data/dataset_pseudo_v2")
THRESHOLD = 0.99

YOLO_DS = Path("data/ikariam_pirate_captcha_dataset")
REVIEWED = Path("data/real_samples_reviewed")
PSEUDO_CSV = Path("pseudo_labels.csv")

LABEL_REMAP = {"j6pj3f_0041.png": "jgpj3f"}
EXCLUDE = {"qw7b9m_0032.png"}

CHARSET = set("abcdefghjklmnpqrstuvwxy23457")


def main():
    for split in ("train", "val"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "text_labels" / split).mkdir(parents=True, exist_ok=True)

    yolo_train = 0
    for img in (YOLO_DS / "images/train").glob("*.png"):
        shutil.copy2(img, OUT / "images/train" / img.name)
        lbl = YOLO_DS / "text_labels/train" / (img.stem + ".txt")
        if lbl.exists():
            shutil.copy2(lbl, OUT / "text_labels/train" / lbl.name)
        yolo_train += 1
    print(f"  YOLO train: {yolo_train}")

    yolo_val = 0
    for img in (YOLO_DS / "images/val").glob("*.png"):
        shutil.copy2(img, OUT / "images/val" / img.name)
        lbl = YOLO_DS / "text_labels/val" / (img.stem + ".txt")
        if lbl.exists():
            shutil.copy2(lbl, OUT / "text_labels/val" / lbl.name)
        yolo_val += 1
    print(f"  YOLO val: {yolo_val}")

    reviewed = 0
    for img in sorted(REVIEWED.glob("*.png")):
        if img.name in EXCLUDE:
            continue
        label = LABEL_REMAP.get(img.name)
        if label is None:
            stem = img.stem
            parts = stem.rsplit("_", 1)
            label = parts[0] if len(parts) == 2 and parts[1].isdigit() else stem
        if not all(c in CHARSET for c in label):
            continue
        dst_name = f"reviewed_{img.name}"
        shutil.copy2(img, OUT / "images/train" / dst_name)
        (OUT / "text_labels/train" / f"reviewed_{img.stem}.txt").write_text(label.upper() + "\n")
        reviewed += 1
    print(f"  Reviewed: {reviewed}")

    pseudo = 0
    pseudo_skipped = 0
    with open(PSEUDO_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            conf = float(row["confidence"])
            if conf < THRESHOLD:
                pseudo_skipped += 1
                continue
            label = row["predicted_label"].lower()
            if not label or not all(c in CHARSET for c in label):
                pseudo_skipped += 1
                continue
            src_path = Path(row["image_path"])
            if not src_path.exists():
                pseudo_skipped += 1
                continue
            if "real_samples_unlabeled" in str(src_path):
                prefix = "ul_"
            elif "/real_samples/" in str(src_path):
                prefix = "rs_"
            else:
                prefix = "ps_"
            dst_name = f"{prefix}{src_path.name}"
            shutil.copy2(src_path, OUT / "images/train" / dst_name)
            (OUT / "text_labels/train" / f"{prefix}{src_path.stem}.txt").write_text(label.upper() + "\n")
            pseudo += 1
    print(f"  Pseudo-labeled (>={THRESHOLD}): {pseudo}  (skipped {pseudo_skipped})")

    total_train = len(list((OUT / "images/train").glob("*.png")))
    total_val = len(list((OUT / "images/val").glob("*.png")))
    print(f"\nTotal: {total_train} train, {total_val} val")
    print(f"Output: {OUT}")


if __name__ == "__main__":
    main()
