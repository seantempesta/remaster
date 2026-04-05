"""
Measure sharpness metrics for all training data targets.

Computes per-frame:
- Laplacian variance (second derivative, measures fine detail)
- Sobel gradient magnitude mean (first derivative, what edge-aware blur uses)
- Resolution (width x height)
- Source prefix

Saves results as a CSV and pickled DataFrame for analysis.

Usage:
    python tools/measure_sharpness.py
    python tools/measure_sharpness.py --data-dir data/mixed_pairs --split train
    python tools/measure_sharpness.py --data-dir data/mixed_val --split val
"""
import os
import sys
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def classify_source(filename):
    """Extract source name from filename prefix."""
    if filename.startswith("synth_"):
        parts = filename.split("_")
        return f"{parts[0]}_{parts[1]}"
    return filename.split("_")[0]


def measure_frame(img_path):
    """Compute sharpness metrics for a single image.

    Returns dict with laplacian_var, sobel_mean, width, height.
    """
    img = cv2.imread(img_path)
    if img is None:
        return None

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Laplacian variance (on uint8 -> CV_64F)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = lap.var()

    gray = gray.astype(np.float32) / 255.0

    # Sobel gradient magnitude (same as edge_aware_degrade uses)
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    sobel_mean = sobel_mag.mean()

    return {
        "laplacian_var": float(lap_var),
        "sobel_mean": float(sobel_mean),
        "width": w,
        "height": h,
    }


def main():
    parser = argparse.ArgumentParser(description="Measure sharpness of training data targets")
    parser.add_argument("--data-dir", type=str, default="data/training/train",
                        help="Data directory containing target/ subdir")
    parser.add_argument("--split", type=str, default="train",
                        help="Split name for output filename (train or val)")
    parser.add_argument("--output-dir", type=str, default="data/analysis",
                        help="Output directory for CSV and pickle")
    args = parser.parse_args()

    target_dir = os.path.join(args.data_dir, "target")
    if not os.path.isdir(target_dir):
        print(f"ERROR: {target_dir} not found")
        sys.exit(1)

    files = sorted(os.listdir(target_dir))
    print(f"Measuring {len(files)} targets in {target_dir}")

    rows = []
    t0 = time.time()

    for i, filename in enumerate(files):
        filepath = os.path.join(target_dir, filename)
        metrics = measure_frame(filepath)
        if metrics is None:
            print(f"  WARN: could not read {filename}")
            continue

        metrics["filename"] = filename
        metrics["source"] = classify_source(filename)
        rows.append(metrics)

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            fps = (i + 1) / elapsed
            print(f"  {i+1}/{len(files)} ({fps:.0f} fps)")

    elapsed = time.time() - t0
    print(f"  Done: {len(rows)} frames in {elapsed:.1f}s ({len(rows)/elapsed:.0f} fps)")

    df = pd.DataFrame(rows)

    # Summary
    print(f"\n{'Source':<20} {'Count':>6} {'Lap Mean':>10} {'Lap Med':>10} {'Sobel Mean':>12} {'Sobel Med':>12}")
    print("-" * 75)
    for source in sorted(df["source"].unique()):
        sub = df[df["source"] == source]
        print(f"{source:<20} {len(sub):>6} "
              f"{sub['laplacian_var'].mean():>10.1f} {sub['laplacian_var'].median():>10.1f} "
              f"{sub['sobel_mean'].mean():>12.4f} {sub['sobel_mean'].median():>12.4f}")

    print(f"\n{'ALL':<20} {len(df):>6} "
          f"{df['laplacian_var'].mean():>10.1f} {df['laplacian_var'].median():>10.1f} "
          f"{df['sobel_mean'].mean():>12.4f} {df['sobel_mean'].median():>12.4f}")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, f"sharpness_{args.split}.csv")
    pkl_path = os.path.join(args.output_dir, f"sharpness_{args.split}.pkl")
    df.to_csv(csv_path, index=False)
    df.to_pickle(pkl_path)
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {pkl_path}")
    print(f"\nLoad in Python: df = pd.read_pickle('{pkl_path}')")


if __name__ == "__main__":
    main()
