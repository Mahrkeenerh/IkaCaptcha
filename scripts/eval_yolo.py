"""Evaluate the IkabotAPI YOLOv8n ONNX model on the original YOLO dataset."""

import os
import glob
import numpy as np
import cv2
import onnxruntime as ort

# Class mapping from the upstream IkabotAPI source
CLASS_NAMES = {
    0: 'B', 1: '2', 2: 'D', 3: 'X', 4: '5', 5: 'M', 6: 'W', 7: 'A',
    8: '7', 9: '4', 10: 'N', 11: 'L', 12: 'P', 13: 'V', 14: 'J', 15: 'H',
    16: 'C', 17: '3', 18: 'U', 19: 'Q', 20: 'Y', 21: 'S', 22: 'T', 23: 'K',
    24: 'R', 25: 'E', 26: 'G', 27: 'F'
}

MODEL_PATH = "models/yolov8n-ikariam-pirates-mAP-0_989.onnx"
DATASET_DIR = "data/ikariam_pirate_captcha_dataset"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45


def load_model(path):
    session = ort.InferenceSession(path)
    inp = session.get_inputs()[0]
    print(f"Model input: {inp.name}, shape: {inp.shape}, dtype: {inp.type}")
    out = session.get_outputs()[0]
    print(f"Model output: {out.name}, shape: {out.shape}")
    return session


def preprocess(img, input_size=(640, 640)):
    """Letterbox + normalize for YOLOv8."""
    h, w = img.shape[:2]
    scale = min(input_size[0] / h, input_size[1] / w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((input_size[0], input_size[1], 3), 114, dtype=np.uint8)
    top = (input_size[0] - nh) // 2
    left = (input_size[1] - nw) // 2
    canvas[top:top+nh, left:left+nw] = resized
    blob = canvas.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis]
    return blob, scale, left, top


def postprocess(output, scale, pad_left, pad_top, orig_w, orig_h,
                conf_thresh=CONF_THRESHOLD, iou_thresh=IOU_THRESHOLD):
    pred = output[0].T
    boxes = pred[:, :4]
    scores = pred[:, 4:]
    max_scores = scores.max(axis=1)
    mask = max_scores > conf_thresh
    boxes = boxes[mask]
    scores = scores[mask]
    max_scores = max_scores[mask]
    class_ids = scores.argmax(axis=1)
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    x1 = (x1 - pad_left) / scale
    y1 = (y1 - pad_top) / scale
    x2 = (x2 - pad_left) / scale
    y2 = (y2 - pad_top) / scale
    indices = cv2.dnn.NMSBoxes(
        [(float(x1[i]), float(y1[i]), float(x2[i]-x1[i]), float(y2[i]-y1[i]))
         for i in range(len(x1))],
        max_scores.tolist(), conf_thresh, iou_thresh
    )
    if len(indices) == 0:
        return []
    indices = indices.flatten()
    detections = [(float(x1[i]), class_ids[i], float(max_scores[i])) for i in indices]
    detections.sort(key=lambda d: d[0])
    return detections


def parse_yolo_label(label_path, img_w, img_h):
    chars = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            cls_id = int(parts[0])
            cx = float(parts[1])
            chars.append((cx, cls_id))
    chars.sort(key=lambda c: c[0])
    return [CLASS_NAMES[c[1]] for c in chars]


def evaluate(split="val"):
    session = load_model(MODEL_PATH)
    input_name = session.get_inputs()[0].name

    img_dir = os.path.join(DATASET_DIR, "images", split)
    lbl_dir = os.path.join(DATASET_DIR, "labels", split)

    images = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    print(f"\nEvaluating on {split}: {len(images)} images\n")

    total = 0
    full_correct = 0
    char_correct = 0
    char_total = 0
    errors = []

    for img_path in images:
        name = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(lbl_dir, name + ".txt")
        if not os.path.exists(lbl_path):
            continue

        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        gt_chars = parse_yolo_label(lbl_path, w, h)
        gt_str = "".join(gt_chars).upper()

        blob, scale, pad_left, pad_top = preprocess(img)
        output = session.run(None, {input_name: blob})[0]
        dets = postprocess(output, scale, pad_left, pad_top, w, h)
        pred_chars = [CLASS_NAMES[d[1]] for d in dets]
        pred_str = "".join(pred_chars).upper()

        total += 1
        if pred_str == gt_str:
            full_correct += 1
        else:
            errors.append((name, gt_str, pred_str))

        for i, gc in enumerate(gt_chars):
            char_total += 1
            if i < len(pred_chars) and pred_chars[i].upper() == gc.upper():
                char_correct += 1

    print(f"Full captcha accuracy: {full_correct}/{total} = {100*full_correct/total:.1f}%")
    print(f"Character accuracy:    {char_correct}/{char_total} = {100*char_correct/char_total:.1f}%")
    print(f"\nFirst 20 errors:")
    for name, gt, pred in errors[:20]:
        print(f"  {name}: GT={gt}  PRED={pred}")

    return full_correct / total if total > 0 else 0


if __name__ == "__main__":
    evaluate("val")
    print("\n" + "="*50 + "\n")
    evaluate("train")
