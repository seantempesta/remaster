"""
Visualize sharpness distribution of training data.

Produces:
1. Distribution plots (histogram + box plot) for Laplacian and Sobel by source
2. Sample grid showing input/target pairs at different sharpness percentiles

Requires: sharpness_train.pkl from measure_sharpness.py

Usage:
    python tools/visualize_sharpness.py
    python tools/visualize_sharpness.py --data-dir data/mixed_pairs --pkl data/sharpness_train.pkl
"""
import os
import sys
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_distributions(df, output_dir):
    """Plot sharpness distributions by source."""
    sources = sorted(df["source"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(sources)))
    source_colors = dict(zip(sources, colors))

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Laplacian histogram
    ax = axes[0, 0]
    for src in sources:
        sub = df[df["source"] == src]
        ax.hist(sub["laplacian_var"], bins=50, alpha=0.6, label=src,
                color=source_colors[src], density=True)
    ax.set_xlabel("Laplacian Variance")
    ax.set_ylabel("Density")
    ax.set_title("Laplacian Variance Distribution")
    ax.legend(fontsize=8)
    ax.set_xlim(0, df["laplacian_var"].quantile(0.95))

    # Laplacian box plot
    ax = axes[0, 1]
    data = [df[df["source"] == src]["laplacian_var"].values for src in sources]
    bp = ax.boxplot(data, labels=sources, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("Laplacian Variance")
    ax.set_title("Laplacian Variance by Source")
    ax.tick_params(axis='x', rotation=30)

    # Sobel histogram
    ax = axes[1, 0]
    for src in sources:
        sub = df[df["source"] == src]
        ax.hist(sub["sobel_mean"], bins=50, alpha=0.6, label=src,
                color=source_colors[src], density=True)
    ax.set_xlabel("Sobel Mean Gradient")
    ax.set_ylabel("Density")
    ax.set_title("Sobel Mean Gradient Distribution")
    ax.legend(fontsize=8)
    ax.set_xlim(0, df["sobel_mean"].quantile(0.95))

    # Sobel box plot
    ax = axes[1, 1]
    data = [df[df["source"] == src]["sobel_mean"].values for src in sources]
    bp = ax.boxplot(data, labels=sources, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("Sobel Mean Gradient")
    ax.set_title("Sobel Mean Gradient by Source")
    ax.tick_params(axis='x', rotation=30)

    plt.tight_layout()
    path = os.path.join(output_dir, "sharpness_distributions.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_scatter(df, output_dir):
    """Scatter plot: Laplacian vs Sobel colored by source."""
    sources = sorted(df["source"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(sources)))

    fig, ax = plt.subplots(figsize=(12, 8))
    for src, color in zip(sources, colors):
        sub = df[df["source"] == src]
        ax.scatter(sub["sobel_mean"], sub["laplacian_var"],
                   alpha=0.3, s=10, color=color, label=src)

    ax.set_xlabel("Sobel Mean Gradient")
    ax.set_ylabel("Laplacian Variance")
    ax.set_title("Sharpness: Laplacian vs Sobel (each dot = one frame)")
    ax.legend(markerscale=3)

    # Clip to 95th percentile for readability
    ax.set_xlim(0, df["sobel_mean"].quantile(0.98))
    ax.set_ylim(0, df["laplacian_var"].quantile(0.98))

    plt.tight_layout()
    path = os.path.join(output_dir, "sharpness_scatter.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def make_sample_grid(df, data_dir, output_dir, metric="laplacian_var",
                     percentiles=(5, 25, 50, 75, 95), crop_size=384):
    """Create a grid showing input/target pairs at different sharpness percentiles.

    Each row = one percentile. Shows 3 samples near that percentile.
    Left half = input (degraded), right half = target (clean).
    """
    thresholds = [np.percentile(df[metric], p) for p in percentiles]
    samples_per_row = 3

    rows = []
    for pct, thresh in zip(percentiles, thresholds):
        # Find frames nearest to this percentile value
        df_sorted = df.iloc[(df[metric] - thresh).abs().argsort()]
        picked = df_sorted.head(samples_per_row * 3)
        # Spread picks across sources if possible
        picked = picked.drop_duplicates(subset="source").head(samples_per_row)
        if len(picked) < samples_per_row:
            picked = df_sorted.head(samples_per_row)

        row_images = []
        for _, row in picked.iterrows():
            inp_path = os.path.join(data_dir, "input", row["filename"])
            tgt_path = os.path.join(data_dir, "target", row["filename"])

            inp = cv2.imread(inp_path)
            tgt = cv2.imread(tgt_path)
            if inp is None or tgt is None:
                continue

            # Center crop
            h, w = inp.shape[:2]
            cy, cx = h // 2, w // 2
            half = crop_size // 2
            inp_crop = inp[cy-half:cy+half, cx-half:cx+half]
            tgt_crop = tgt[cy-half:cy+half, cx-half:cx+half]

            # Side by side with thin separator
            sep = np.full((crop_size, 4, 3), 255, dtype=np.uint8)
            pair = np.hstack([inp_crop, sep, tgt_crop])
            row_images.append(pair)

        if row_images:
            # Add label
            label_img = np.zeros((40, row_images[0].shape[1] * len(row_images) + 8 * (len(row_images) - 1), 3), dtype=np.uint8)
            label = f"P{pct} ({metric}={thresh:.1f})"
            cv2.putText(label_img, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Horizontal join samples
            hsep = np.full((crop_size, 8, 3), 128, dtype=np.uint8)
            combined = row_images[0]
            for img in row_images[1:]:
                combined = np.hstack([combined, hsep, img])

            rows.append(label_img)
            rows.append(combined)
            rows.append(np.full((8, combined.shape[1], 3), 64, dtype=np.uint8))

    if not rows:
        print("No samples to show")
        return

    # Pad all rows to same width
    max_w = max(r.shape[1] for r in rows)
    padded = []
    for r in rows:
        if r.shape[1] < max_w:
            pad = np.zeros((r.shape[0], max_w - r.shape[1], 3), dtype=np.uint8)
            r = np.hstack([r, pad])
        padded.append(r)

    grid = np.vstack(padded)
    path = os.path.join(output_dir, f"sharpness_samples_{metric}.png")
    cv2.imwrite(path, grid)
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize sharpness distributions")
    parser.add_argument("--pkl", type=str, default="data/analysis/sharpness_train.pkl",
                        help="Path to sharpness pickle file")
    parser.add_argument("--data-dir", type=str, default="data/training/train",
                        help="Data directory with input/ and target/ subdirs")
    parser.add_argument("--output-dir", type=str, default="data/analysis",
                        help="Output directory for plots")
    args = parser.parse_args()

    if not os.path.exists(args.pkl):
        print(f"ERROR: {args.pkl} not found. Run measure_sharpness.py first.")
        sys.exit(1)

    df = pd.read_pickle(args.pkl)
    print(f"Loaded {len(df)} frames from {args.pkl}")
    print(f"Sources: {sorted(df['source'].unique())}")

    os.makedirs(args.output_dir, exist_ok=True)
    plot_distributions(df, args.output_dir)
    plot_scatter(df, args.output_dir)
    make_sample_grid(df, args.data_dir, args.output_dir, metric="laplacian_var")
    make_sample_grid(df, args.data_dir, args.output_dir, metric="sobel_mean")

    print("\nDone. Files in", args.output_dir)


if __name__ == "__main__":
    main()
