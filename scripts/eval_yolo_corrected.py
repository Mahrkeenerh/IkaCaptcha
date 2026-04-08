"""Evaluate YOLO ONNX model on the corrected val set (text labels)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so we can import eval_yolo

import cv2

from eval_yolo import CLASS_NAMES, load_model, preprocess, postprocess, MODEL_PATH


def main():
    session = load_model(MODEL_PATH)
    input_name = session.get_inputs()[0].name

    img_dir = Path("data/dataset_pseudo_v2/images/val")
    lbl_dir = Path("data/dataset_pseudo_v2/text_labels/val")

    images = sorted(img_dir.glob("*.png"))
    print(f"\nEvaluating YOLO on corrected val: {len(images)} images\n")

    total = 0
    full_correct = 0
    char_correct = 0
    char_total = 0

    for img_path in images:
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue

        gt_str = lbl_path.read_text().strip().upper()

        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        blob, scale, pad_left, pad_top = preprocess(img)
        output = session.run(None, {input_name: blob})[0]
        dets = postprocess(output, scale, pad_left, pad_top, w, h)
        pred_chars = [CLASS_NAMES[d[1]] for d in dets]
        pred_str = "".join(pred_chars).upper()

        total += 1
        if pred_str == gt_str:
            full_correct += 1

        for i, gc in enumerate(gt_str):
            char_total += 1
            if i < len(pred_str) and pred_str[i] == gc:
                char_correct += 1

    print(f"Full sequence accuracy: {full_correct}/{total} = {100*full_correct/total:.1f}%")
    print(f"Character accuracy:    {char_correct}/{char_total} = {100*char_correct/char_total:.1f}%")


if __name__ == "__main__":
    main()
