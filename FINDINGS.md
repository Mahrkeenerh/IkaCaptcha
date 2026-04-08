# iKaptcha — Final Findings

## Results

| Model | Params | Training Data | Original Val (300) | Corrected Val (298) |
|---|---|---|---|---|
| **YOLOv8n (IkabotAPI)** | ~3M | 1200 | 78.7% | 81.2% |
| **Our CRNN (production)** | **1.66M** | **11,210** | **95.0%** | **97.3%** |

**Character accuracy**: 99.5% (CRNN) vs 96.5% (YOLO)
**Absolute improvement**: +16.1% full-sequence accuracy on corrected val

All numbers are from the **final training epoch** (not cherry-picked best checkpoint). Last 10 epochs were stable at 96.7–97.3% on corrected val.

## Architecture (`ikaptcha/model.py`)

CRNN (CNN + BiLSTM + CTC) with ~1.66M parameters:

```
Input: 48×256 RGB
├── ResBlock+SE 3→32    + MaxPool(2,2)   → 24×128
├── ResBlock+SE 32→64   + MaxPool(2,2)   → 12×64  + Dropout2d(0.2)
├── ResBlock+SE 64→128  + MaxPool(2,1)   →  6×64  + Dropout2d(0.2)
├── ResBlock+SE 128→256 + MaxPool(2,1)   →  3×64  + Dropout2d(0.2)
└── AdaptiveAvgPool((1,None))            →  1×64  (T=64 timesteps)

BiLSTM: 1 layer, hidden=128, bidirectional (256-dim output)
FC: Linear(256 → 29) + Dropout(0.4)
CTC loss with 28-char blank-prepended charset
```

Key design choices (verified empirically):
- **SE (Squeeze-and-Excitation) blocks** on every ResBlock — channel attention for discriminating confusable characters (e/c, h/b)
- **MaxPool(2,1) in last two blocks** — preserves horizontal resolution for T=64 timesteps
- **4-block CNN** outperformed 3-block (deeper features resolved mid-image errors)
- **Single-layer BiLSTM** — deeper BiLSTM didn't help on random char sequences
- **CTC loss**, not attention decoder — monotonic alignment is a feature on small data

## Training Pipeline

Two-phase training (`scripts/train.py`), ~90 min total on single GPU:

1. **Pretrain on synthetic (50 epochs)**
   - 65k synthetic samples from `generate_captcha.py`
   - OneCycleLR, max_lr=3e-4, 3-epoch warmup
   - Gets the model to 10-12% on real val (not useful alone, but establishes good features)

2. **Mixed training (40 epochs)**
   - 11,210 real + ~65k synthetic with WeightedRandomSampler (3× real oversample)
   - OneCycleLR, max_lr=2e-4, 10% warmup, smooth cosine decay
   - NO warm restarts, NO SWA (both counterproductive with CTC)
   - Gradient clip norm 5.0, weight decay 1e-2

Augmentation: RandomAffine(±5°), ColorJitter, GaussianBlur(p=0.3), GaussianNoise, RandomErasing. Conservative — ViT-style heavy augmentation hurts CTC alignment.

## Data Journey

| Source | Count | Notes |
|---|---|---|
| YOLO dataset train | 1,200 | From IkabotAPI authors |
| Hand-reviewed | 99 | Manually labeled, 1 excluded for ambiguity |
| Pseudo-labeled (≥0.99 confidence) | 9,911 | Auto-labeled with earlier model, confidence-filtered |
| **Total real train** | **11,210** | |
| YOLO val | 300 | Untouched for comparison vs YOLO baseline |
| Corrected val | 298 | 7 labels corrected, 2 discarded (found via k-fold + review) |
| Synthetic | ~65,000 | From `scripts/generate_captcha.py`, 28-char charset |

**Captcha fetching**: Server rate-limits aggressively. Safe delay is 1.0–2.5s randomized. Fetching >1300 at <0.5s delay triggers IP block (~24h). Cookie is `ikariam=...`, from browser dev tools.

## What Worked

1. **Synthetic pretraining** (biggest lever by far — went from 45% to 87% on first try)
2. **Mixed training with weighted sampling** (real as "anchor", synthetic as bulk)
3. **K-fold cross-validation for label quality** — found 11 mislabeled training samples, 7 mislabeled val samples. Each fixed val label = +0.33%.
4. **Confidence-filtered pseudo-labeling** — 9.9k extra real samples at ≥0.99 confidence. Biggest single jump (94.3% → 97.3%).
5. **SE (Squeeze-Excitation) blocks** — channel attention. 87% → 94% jump.
6. **OneCycleLR with no warm restarts/SWA** — CTC is fragile, smooth schedules win.
7. **Weight decay 1e-2** (not 1e-4) — much stronger regularization for small data.

## What Didn't Work

1. **ViT from scratch** — 71% (vs CRNN's 97%). Lacks inductive bias at this data scale.
2. **Pretrained PARSeq (23.8M params)** — 59%, then overfit. Domain gap too large; params too many for 11k real samples.
3. **Test-time augmentation (TTA)** — net negative. Augmentations corrupt CTC alignment.
4. **Beam search** — identical to greedy. CTC posteriors were already sharp.
5. **Checkpoint ensemble** — later checkpoints hurt the average.
6. **SWA (Stochastic Weight Averaging)** — 92% vs 94% best checkpoint. Warm restarts corrupted the averaging.
7. **Warm restarts (CosineAnnealingWarmRestarts)** — LR spikes destroyed CTC alignments, cost ~20 epochs to recover each time.
8. **Auxiliary length prediction head** — consultant said length errors are segmentation issues, not counting.

## Verified Character Set (28 chars)

`ABCDEFGHJKLMNPQRSTUVWXY23457`

Missing: **I, O, Z, 0, 1, 6, 8, 9** — excluded by game server because visually ambiguous. Verified via manual review of 100+ samples and analysis of 1500 YOLO-labeled samples.

## Error Analysis (8 remaining errors on 298 val)

All long-tail single-character confusions:
- `g↔c`, `e↔c`, `r↔p`, `v↔7`, `q↔b` (single-char swaps)
- A few likely label errors with length mismatches

Model confidence on these is high (confidently wrong), suggesting they're either label noise or truly ambiguous characters that humans would also struggle with. 97% is close to the noise floor for this task.

## Production Model

- **PyTorch**: `models/final_mixed.pth` — CRNN class in `ikaptcha/model.py` with `hidden_size=128`
- **ONNX**: `models/crnn.onnx` — exported via `scripts/export_onnx.py`, 6.4 MB, opset 18
- **Input**: 48×256 RGB, normalized with mean/std [0.5, 0.5, 0.5]
- **Output**: T=64 timesteps × 29 classes (greedy CTC decode)
- **Inference**: `scripts/predict.py` (PyTorch) or `scripts/predict_onnx.py` (ONNX, no torch dep)
- **Runtime compatibility**: loads cleanly via `onnxruntime` and via OpenCV's `cv2.dnn` (≥4.8.0). Bit-exact predictions vs PyTorch on every val sample across all three backends.

### ONNX Export & Parity

`scripts/export_onnx.py` exports `models/final_mixed.pth` to `models/crnn.onnx` and verifies prediction parity against PyTorch on both validation sets. Result on all validation samples: **0 string mismatches, identical 95.0% / 97.3% accuracy**.

Two non-obvious export details (both handled in the script):
1. The model's `AdaptiveAvgPool2d((1, None))` is rejected by the legacy ONNX exporter (output_size contains None). Swapped for a fixed `AvgPool2d((3, 1))` since H=3 at that stage — verified mathematically equivalent.
2. The dynamo exporter (`torch.onnx.export(..., dynamo=True)`) constant-folded the batch dim into a Reshape, breaking dynamic batching. Used the legacy TorchScript exporter (`dynamo=False`) instead.

### Quantization Experiment (rejected)

Dynamic int8 weight quantization produced a 1.6 MB model (3.9× smaller) with identical accuracy on val and only 2/11,210 prediction disagreements vs fp32 on the full real pool. **But** the model is LSTM-dominated and onnxruntime's CPU LSTM kernel has no efficient int8 path — int8 was *slower* (68 vs 197 img/s) due to dequant/requant overhead. Combined with poor `onnxruntime-web` WebGPU support for int8, we ship fp32. The 4.8 MB savings on a one-time download isn't worth the runtime risk.
