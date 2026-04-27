"""
Parity test: NumPy CRNN vs ONNX on the 9 hand-labeled samples.

Same shape as the pure-Python test so the two are directly comparable.
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
from inference import forward, ctc_greedy_decode, load_weights


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

    print("Loading NumPy weights...")
    weights = load_weights()

    rows = []
    np_times_warm = []
    onnx_times = []

    for fname, label in SAMPLE_LABELS.items():
        path = samples_dir / fname
        img = Image.open(path)
        x_np = preprocess_pil(img)                    # (1, 3, 48, 256) float32

        # ONNX
        t0 = time.perf_counter()
        logits_onnx = session.run(["logits"], {"input": x_np})[0]
        t_onnx = time.perf_counter() - t0
        onnx_times.append(t_onnx)
        pred_onnx = greedy_decode_numpy(logits_onnx)[0]

        # NumPy: same preprocessed input, drop the batch dim.
        x_chw = x_np[0]
        # Time best-of-3 to get a stable measurement (NumPy first call has
        # some BLAS / allocation overhead).
        t_runs = []
        for _ in range(3):
            t0 = time.perf_counter()
            logits_np = forward(x_chw, weights)
            pred_np = ctc_greedy_decode(logits_np)
            t_runs.append(time.perf_counter() - t0)
        t_np = min(t_runs)
        np_times_warm.append(t_np)

        onnx_arr = logits_onnx[:, 0, :].astype(np.float64)
        np_arr = logits_np.astype(np.float64)
        max_diff = float(np.max(np.abs(onnx_arr - np_arr)))

        match = "OK" if pred_onnx == pred_np else "DIFF"
        label_match = "GT-OK" if pred_np == label else "GT-MISS"
        rows.append((fname, label, pred_onnx, pred_np, max_diff, t_np, t_onnx, match, label_match))

        print(f"  {fname:12s}  label={label:9s}  onnx={pred_onnx:10s}  "
              f"np={pred_np:10s}  Δlogit={max_diff:.3e}  "
              f"np={t_np*1000:6.1f}ms  onnx={t_onnx*1000:5.1f}ms  {match}  {label_match}")

    print("\n" + "=" * 70)
    n = len(rows)
    np_mean = sum(np_times_warm) / n
    onnx_mean = sum(onnx_times) / n
    matches = sum(1 for r in rows if r[7] == "OK")
    label_matches = sum(1 for r in rows if r[8] == "GT-OK")
    max_logit_diff = max(r[4] for r in rows)

    print(f"NumPy vs ONNX string match:        {matches}/{n}")
    print(f"NumPy vs ground truth:             {label_matches}/{n}")
    print(f"Max absolute logit diff:           {max_logit_diff:.3e}")
    print(f"NumPy mean time per image (warm):  {np_mean*1000:.2f} ms")
    print(f"ONNX mean time per image:          {onnx_mean*1000:.2f} ms")
    print(f"Slowdown vs ONNX:                  {np_mean / onnx_mean:.1f}x")

    val_n = 300 + 298
    print(f"\nProjected full-val ({val_n} samples) runtime:")
    print(f"  NumPy: {np_mean * val_n:.2f} s")
    print(f"  ONNX:  {onnx_mean * val_n:.2f} s")


if __name__ == "__main__":
    main()
