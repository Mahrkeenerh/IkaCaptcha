# iKaptcha

A CRNN that solves the Ikariam pirate fortress captcha.

| Model | Params | Original val | Corrected val |
|---|---|---|---|
| YOLOv8n (IkabotAPI baseline) | ~3 M | 78.7% | 81.2% |
| **CRNN (this repo)** | **1.66 M** | **95.0%** | **97.3%** |

Character accuracy: **99.5%**. Production model: `models/crnn.onnx` (6.4 MB).

## Quick start

```bash
uv venv && uv pip install -e .
python scripts/predict_onnx.py data/samples/test1.png data/samples/test2.png
```

```
data/samples/test1.png: b45d5eee  (conf=0.993)
data/samples/test2.png: aqrckd3   (conf=0.995)
```

`scripts/predict_onnx.py` depends only on `onnxruntime`, `Pillow`, and `numpy` — no PyTorch needed at inference time. Use it as the reference implementation when porting to other runtimes (browser via `onnxruntime-web`, mobile, OpenCV's `cv2.dnn`, etc).

## Character set

The captcha uses **only 28 characters**, not 36. The game server excludes visually ambiguous ones:

```
Letters (24): A B C D E F G H J K L M N P Q R S T U V W X Y
Digits  (4):  2 3 4 5 7
Excluded:     0 1 6 8 9 I O Z
```

## Repo layout

```
ikaptcha/                        importable package (CRNN, charset, decoders, transforms)
scripts/                         entry-point scripts
├── predict.py predict_onnx.py   single-image inference (PyTorch / ONNX)
├── train.py                     two-phase training
├── export_onnx.py               PyTorch -> ONNX with parity verification
├── eval_compare.py              eval against both val sets
├── eval_yolo.py                 YOLO baseline
├── generate_captcha.py          synthetic captcha generator
├── generate_dataset.py          multiprocess synthetic dataset builder
├── fetch_captchas.py            real captcha fetcher
├── pseudo_label.py              confidence-scored auto-labeling
├── prepare_pseudo_train_v2.py   builds the production dataset
└── kfold_validate.py            label-quality cross-validation
data/
├── samples/                     9 hand-labeled test images
├── ikariam_pirate_captcha_dataset/   original 1,200/300 YOLO dataset
└── dataset_pseudo_v2/           production 11,210/298 train/val
models/
├── crnn.onnx                    production ONNX (ship this)
├── final_mixed.pth              same weights, PyTorch format
└── yolov8n-...onnx              YOLO baseline for comparison
fonts/                           ~120 .ttf files for synthetic generation
```

## Credits

YOLOv8n baseline and the original 1,500-sample dataset are from [IkabotAPI](https://github.com/Ikabot-Collective/IkabotAPI).

Full technical writeup, ablations, and "what didn't work" notes: see `FINDINGS.md`.
