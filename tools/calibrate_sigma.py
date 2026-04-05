"""
Calibrate per-frame denoising sigma based on noise level.

Two modes:
  1. Default: bucket frames by noise, pre-render sigma comparison grids
  2. --fit: after human labeling, fit piecewise-linear noise->sigma curve

Usage:
    # Pre-render sigma comparison grids for labeling
    python tools/calibrate_sigma.py

    # After labeling with label_sigma.py, fit the curve
    python tools/calibrate_sigma.py --fit

    # Custom number of samples per bucket
    python tools/calibrate_sigma.py --samples-per-bucket 8
"""
import sys
import os
import gc
import pickle
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_DIR = Path("data")
ORIGINALS_DIR = DATA_DIR / "originals"
CALIBRATION_DIR = DATA_DIR / "calibration"
GRIDS_DIR = CALIBRATION_DIR / "grids"

# Noise level buckets and sigma options to test
NOISE_BUCKETS = [(0, 2), (2, 4), (4, 7), (7, 10), (10, 15), (15, 25), (25, 50), (50, 200)]
SIGMA_OPTIONS = [0, 1, 2, 3, 5, 8, 10, 15]
CROP_SIZE = 512  # center crop for detail visibility


def load_denoiser(device="cuda"):
    """Load pretrained DRUNet Gaussian denoiser."""
    from lib.paths import resolve_kair_dir
    kair_dir = resolve_kair_dir()
    sys.path.insert(0, str(kair_dir))
    from models.network_unet import UNetRes

    weights_path = os.path.join(str(kair_dir), "model_zoo", "drunet_color.pth")
    model = UNetRes(in_nc=4, out_nc=3, nc=[64, 128, 256, 512], nb=4,
                    act_mode='R', bias=False)
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt, strict=True)
    del ckpt; gc.collect()
    model = model.half().to(device).eval()
    print(f"Denoiser loaded ({torch.cuda.memory_allocated()/1024**2:.0f}MB VRAM)")
    return model


@torch.no_grad()
def denoise_frame(model, frame_bgr, sigma, device="cuda"):
    """Denoise a BGR frame using DRUNet."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    tensor = tensor.half().to(device)

    _, _, h, w = tensor.shape
    noise_map = torch.full((1, 1, h, w), sigma / 255.0, dtype=tensor.dtype, device=device)
    tensor = torch.cat([tensor, noise_map], dim=1)

    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    if pad_h or pad_w:
        tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')

    output = model(tensor)

    if pad_h or pad_w:
        output = output[:, :, :h, :w]

    output = output.squeeze(0).clamp(0, 1).float().cpu().numpy()
    output = (output * 255.0).round().astype(np.uint8)
    output = np.transpose(output, (1, 2, 0))
    return cv2.cvtColor(output, cv2.COLOR_RGB2BGR)


def center_crop(img, size):
    """Center crop an image to size x size."""
    h, w = img.shape[:2]
    cy, cx = h // 2, w // 2
    half = size // 2
    y1 = max(0, cy - half)
    x1 = max(0, cx - half)
    return img[y1:y1 + size, x1:x1 + size]


def do_prerender(args):
    """Bucket frames by noise level, render sigma comparisons."""
    meta_path = ORIGINALS_DIR / "meta.pkl"
    if not meta_path.exists():
        print("ERROR: Run build_training_data.py --extract-only first")
        sys.exit(1)

    df = pd.read_pickle(meta_path)
    print(f"Loaded {len(df)} frames")

    os.makedirs(GRIDS_DIR, exist_ok=True)

    # Bucket frames by noise level
    samples = []
    for lo, hi in NOISE_BUCKETS:
        bucket = df[(df["noise_level"] >= lo) & (df["noise_level"] < hi)]
        if len(bucket) == 0:
            print(f"  Bucket [{lo}-{hi}): empty")
            continue
        # Pick diverse samples (spread across sources)
        picked = bucket.sample(
            n=min(args.samples_per_bucket, len(bucket)),
            random_state=42,
        )
        for _, row in picked.iterrows():
            samples.append({
                "filename": row["filename"],
                "noise_level": row["noise_level"],
                "bucket": f"{lo}-{hi}",
                "source": row["source"],
            })
        print(f"  Bucket [{lo}-{hi}): {len(bucket)} frames, picked {len(picked)}")

    print(f"\nTotal samples: {len(samples)}")

    # Load denoiser and pre-render all sigma options
    model = load_denoiser()

    for i, sample in enumerate(samples):
        filename = sample["filename"]
        orig_path = ORIGINALS_DIR / filename
        frame = cv2.imread(str(orig_path))
        if frame is None:
            print(f"  SKIP: {filename}")
            continue

        # Render each sigma option (center-cropped for detail)
        crop = center_crop(frame, CROP_SIZE)
        rendered = {"original": crop.copy()}
        for sigma in SIGMA_OPTIONS:
            denoised = denoise_frame(model, frame, sigma)
            rendered[f"sigma_{sigma}"] = center_crop(denoised, CROP_SIZE)

        # Save individual crops for Streamlit
        sample_dir = GRIDS_DIR / f"sample_{i:03d}"
        os.makedirs(sample_dir, exist_ok=True)

        for name, img in rendered.items():
            cv2.imwrite(str(sample_dir / f"{name}.png"), img)

        # Also build a combined comparison grid for quick review
        _build_grid(rendered, sample_dir / "grid.png",
                    f"noise={sample['noise_level']:.1f} src={sample['source']}")

        samples[i]["grid_dir"] = str(sample_dir)
        print(f"  [{i+1}/{len(samples)}] {filename} noise={sample['noise_level']:.1f}")

    # Save sample info
    samples_path = CALIBRATION_DIR / "samples.pkl"
    with open(samples_path, "wb") as f:
        pickle.dump(samples, f)
    print(f"\nSaved: {samples_path}")
    print(f"Grids: {GRIDS_DIR}")
    print(f"\nNext: streamlit run tools/label_sigma.py")

    del model
    gc.collect()
    torch.cuda.empty_cache()


def _build_grid(rendered, output_path, title=""):
    """Build a side-by-side comparison grid from rendered crops."""
    imgs = []
    labels = []
    for name in ["original"] + [f"sigma_{s}" for s in SIGMA_OPTIONS]:
        if name in rendered:
            imgs.append(rendered[name])
            labels.append(name.replace("sigma_", "s=").replace("original", "orig"))

    if not imgs:
        return

    # Add labels on top of each image
    labeled = []
    for img, label in zip(imgs, labels):
        canvas = np.zeros((30 + img.shape[0], img.shape[1], 3), dtype=np.uint8)
        cv2.putText(canvas, label, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        canvas[30:, :] = img
        labeled.append(canvas)

    # Horizontal join with separator
    sep = np.full((labeled[0].shape[0], 2, 3), 128, dtype=np.uint8)
    row = labeled[0]
    for img in labeled[1:]:
        row = np.hstack([row, sep, img])

    # Add title
    if title:
        title_bar = np.zeros((35, row.shape[1], 3), dtype=np.uint8)
        cv2.putText(title_bar, title, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        row = np.vstack([title_bar, row])

    cv2.imwrite(str(output_path), row)


def do_fit(args):
    """Fit GP regression models: (noise, laplacian, sobel) -> (sigma, detail_strength).

    Uses Gaussian Process Regression with multiple input features for smooth,
    uncertainty-aware interpolation across all 7K frames from just 48 labels.
    """
    import pandas as pd
    from collections import Counter
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
    from sklearn.preprocessing import StandardScaler

    labels_path = CALIBRATION_DIR / "labels.pkl"
    if not labels_path.exists():
        print("ERROR: Run label_sigma.py first to create labels")
        sys.exit(1)

    with open(labels_path, "rb") as f:
        labels = pickle.load(f)

    # Load originals meta for feature lookup
    meta_path = ORIGINALS_DIR / "meta.pkl"
    if not meta_path.exists():
        print("ERROR: data/originals/meta.pkl not found")
        sys.exit(1)
    df_meta = pd.read_pickle(meta_path)
    meta_lookup = {row["filename"]: row for _, row in df_meta.iterrows()}

    # Extract labeled data with full features
    valid = [l for l in labels if l.get("chosen_sigma") is not None]
    print(f"Loaded {len(labels)} labels, {len(valid)} valid")

    if len(valid) < 3:
        print(f"ERROR: Need at least 3 labeled points, got {len(valid)}")
        sys.exit(1)

    rows = []
    for l in valid:
        meta = meta_lookup.get(l["filename"])
        if meta is None:
            print(f"  WARN: {l['filename']} not in meta.pkl, skipping")
            continue
        rows.append({
            "noise_level": l["noise_level"],
            "laplacian_var": meta["laplacian_var"],
            "sobel_mean": meta["sobel_mean"],
            "sigma": l["chosen_sigma"],
            "detail_strength": l.get("detail_strength", 0.0),
        })

    df_labels = pd.DataFrame(rows)
    feature_cols = ["noise_level", "laplacian_var", "sobel_mean"]
    X = df_labels[feature_cols].values
    y_sigma = df_labels["sigma"].values.astype(float)
    y_detail = df_labels["detail_strength"].values.astype(float)

    print(f"\nTraining GP on {len(X)} points with features: {feature_cols}")
    print(f"  Sigma range: [{y_sigma.min():.0f}, {y_sigma.max():.0f}]")
    print(f"  Detail range: [{y_detail.min():.2f}, {y_detail.max():.2f}]")

    # Standardize features for GP
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # GP kernel: Matern (smooth but flexible) + white noise (label noise)
    kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1.0)

    gp_sigma = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)
    gp_sigma.fit(X_scaled, y_sigma)
    print(f"\n  GP sigma kernel: {gp_sigma.kernel_}")

    gp_detail = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)
    gp_detail.fit(X_scaled, y_detail)
    print(f"  GP detail kernel: {gp_detail.kernel_}")

    # Predict on all originals
    X_all = df_meta[feature_cols].values
    X_all_scaled = scaler.transform(X_all)

    pred_sigma, pred_sigma_std = gp_sigma.predict(X_all_scaled, return_std=True)
    pred_detail, pred_detail_std = gp_detail.predict(X_all_scaled, return_std=True)

    # Clamp to valid ranges
    pred_sigma = np.clip(pred_sigma, 0, 50)
    pred_detail = np.clip(pred_detail, 0, 2.0)

    print(f"\nPredictions for {len(X_all)} frames:")
    print(f"  Sigma: mean={pred_sigma.mean():.1f}, std={pred_sigma_std.mean():.2f}, "
          f"range=[{pred_sigma.min():.1f}, {pred_sigma.max():.1f}]")
    print(f"  Detail: mean={pred_detail.mean():.2f}, std={pred_detail_std.mean():.2f}, "
          f"range=[{pred_detail.min():.2f}, {pred_detail.max():.2f}]")

    # Check training fit
    pred_train_sigma = gp_sigma.predict(X_scaled)
    pred_train_detail = gp_detail.predict(X_scaled)
    sigma_mae = np.abs(pred_train_sigma - y_sigma).mean()
    detail_mae = np.abs(pred_train_detail - y_detail).mean()
    print(f"\n  Train MAE: sigma={sigma_mae:.2f}, detail={detail_mae:.3f}")

    # Determine most-used detail method
    method_counts = Counter(l.get("detail_method", "high_pass") for l in valid
                            if l.get("detail_strength", 0) > 0)
    detail_method = method_counts.most_common(1)[0][0] if method_counts else "high_pass"
    print(f"\n  Detail method (most used): {detail_method}")
    for m, c in method_counts.most_common():
        print(f"    {m}: {c} labels")

    # Save model
    model_data = {
        "type": "gp",
        "gp_sigma": gp_sigma,
        "gp_detail": gp_detail,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "detail_method": detail_method,
        "raw_labels": labels,
        # Keep np.interp fallback for simple use cases
        "noise_levels": df_labels.sort_values("noise_level")["noise_level"].values,
        "sigmas": df_labels.sort_values("noise_level")["sigma"].values,
        "detail_strengths": df_labels.sort_values("noise_level")["detail_strength"].values,
    }
    model_path = CALIBRATION_DIR / "sigma_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
    print(f"\nSaved: {model_path}")

    # Plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Sigma vs noise_level
        ax = axes[0, 0]
        ax.scatter(df_labels["noise_level"], y_sigma, c='red', s=60, zorder=5, label='Labels')
        ax.scatter(df_meta["noise_level"], pred_sigma, c='blue', s=3, alpha=0.3, label='GP predictions')
        ax.set_xlabel("Noise Level")
        ax.set_ylabel("Sigma")
        ax.set_title("Sigma vs Noise Level")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Detail vs noise_level
        ax = axes[0, 1]
        ax.scatter(df_labels["noise_level"], y_detail, c='red', s=60, zorder=5, label='Labels')
        ax.scatter(df_meta["noise_level"], pred_detail, c='green', s=3, alpha=0.3, label='GP predictions')
        ax.set_xlabel("Noise Level")
        ax.set_ylabel("Detail Strength")
        ax.set_title("Detail vs Noise Level")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Sigma uncertainty
        ax = axes[1, 0]
        ax.hist(pred_sigma_std, bins=50, color='blue', alpha=0.7)
        ax.set_xlabel("GP Sigma Uncertainty (std)")
        ax.set_ylabel("Count")
        ax.set_title(f"Sigma prediction uncertainty (mean={pred_sigma_std.mean():.2f})")

        # Detail uncertainty
        ax = axes[1, 1]
        ax.hist(pred_detail_std, bins=50, color='green', alpha=0.7)
        ax.set_xlabel("GP Detail Uncertainty (std)")
        ax.set_ylabel("Count")
        ax.set_title(f"Detail prediction uncertainty (mean={pred_detail_std.mean():.2f})")

        plt.suptitle(f"GP Calibration: {len(valid)} labels -> {len(df_meta)} predictions", fontsize=14)
        plt.tight_layout()
        plot_path = CALIBRATION_DIR / "sigma_curve.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Plot: {plot_path}")
    except Exception as e:
        print(f"(Plot skipped: {e})")


def main():
    parser = argparse.ArgumentParser(description="Calibrate per-frame denoising sigma")
    parser.add_argument("--fit", action="store_true",
                        help="Fit curve from human labels (run after label_sigma.py)")
    parser.add_argument("--samples-per-bucket", type=int, default=6,
                        help="Number of sample frames per noise bucket")
    args = parser.parse_args()

    if args.fit:
        do_fit(args)
    else:
        do_prerender(args)


if __name__ == "__main__":
    main()
