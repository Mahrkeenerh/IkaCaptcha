"""
Parity test: pure-Python CRNN vs ONNX on the 9 hand-labeled samples.

Reports per-sample string match, max absolute logit diff, and timings.
Projects full-val-set runtime from the per-sample mean.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import onnxruntime as ort
from PIL import Image

from ikaptcha import greedy_decode_numpy, preprocess_pil

sys.path.insert(0, str(Path(__file__).parent))
from inference import forward, ctc_greedy_decode, load_weights, NUM_CLASSES


SAMPLE_LABELS = {
    "test1.png": "b45d5eee", "test2.png": "aqrckd3", "test3.png": "25t352j",
    "test4.png": "5jjpe4b",  "test5.png": "hp5xeqf", "test6.png": "xnwcprl",
    "test7.png": "arh3ml",   "test8.png": "dtmbpw",  "test9.png": "7dxesdjv",
}


def main():
    samples_dir = ROOT / "data" / "samples"
    onnx_path = ROOT / "models" / "ikaptcha.onnx"

    print("Loading ONNX session...")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    print("Loading pure-Python weights...")
    weights = load_weights()

    rows = []
    pp_times = []
    onnx_times = []

    for fname, label in SAMPLE_LABELS.items():
        path = samples_dir / fname
        img = Image.open(path)
        x_np = preprocess_pil(img)  # (1, 3, 48, 256) float32

        # ONNX
        t0 = time.perf_counter()
        logits_onnx = session.run(["logits"], {"input": x_np})[0]  # (T, B, C)
        t_onnx = time.perf_counter() - t0
        onnx_times.append(t_onnx)
        pred_onnx = greedy_decode_numpy(logits_onnx)[0]

        # Pure Python: feed identical input as flat list (CHW order).
        x_flat = x_np[0].reshape(-1).tolist()
        t0 = time.perf_counter()
        logits_pp = forward(x_flat, weights)        # list of 64 lists of 29
        pred_pp = ctc_greedy_decode(logits_pp)
        t_pp = time.perf_counter() - t0
        pp_times.append(t_pp)

        # Compare logits: ONNX shape (T, B, C). Squeeze batch.
        onnx_arr = logits_onnx[:, 0, :]              # (64, 29)
        pp_arr = np.asarray(logits_pp, dtype=np.float64)
        max_diff = float(np.max(np.abs(onnx_arr.astype(np.float64) - pp_arr)))

        match = "OK" if pred_onnx == pred_pp else "DIFF"
        label_match = "GT-OK" if pred_pp == label else "GT-MISS"
        rows.append((fname, label, pred_onnx, pred_pp, max_diff, t_pp, t_onnx, match, label_match))

        print(f"  {fname:12s}  label={label:9s}  onnx={pred_onnx:10s}  "
              f"pp={pred_pp:10s}  Δlogit={max_diff:.3e}  pp={t_pp:6.2f}s  "
              f"onnx={t_onnx*1000:5.1f}ms  {match}  {label_match}")

    print("\n" + "=" * 70)
    n = len(rows)
    pp_mean = sum(pp_times) / n
    onnx_mean = sum(onnx_times) / n
    matches = sum(1 for r in rows if r[7] == "OK")
    label_matches = sum(1 for r in rows if r[8] == "GT-OK")
    max_logit_diff = max(r[4] for r in rows)

    print(f"Pure-Python vs ONNX string match: {matches}/{n}")
    print(f"Pure-Python vs ground truth:      {label_matches}/{n}")
    print(f"Max absolute logit diff:          {max_logit_diff:.3e}")
    print(f"Pure-Python mean time per image:  {pp_mean:.2f} s")
    print(f"ONNX mean time per image:         {onnx_mean*1000:.1f} ms")

    # Project full validation set
    val_n = 300 + 298  # original + corrected
    print(f"\nProjected full-val ({val_n} samples) runtime:")
    print(f"  Pure-Python: {pp_mean * val_n / 60:.1f} min")
    print(f"  ONNX:        {onnx_mean * val_n:.1f} s")


if __name__ == "__main__":
    main()
