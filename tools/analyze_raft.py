"""Analyze RAFT alignment data - interactive exploration."""
import numpy as np
from pathlib import Path
import cv2
import os

base = Path('data/archive/raft_modal_extract')


def load_center_data(model, center_idx):
    """Load center frame, all warped neighbors, and masks."""
    d = base / model
    center = cv2.imread(str(d / f'originals/original_{center_idx:06d}.png'))
    center = cv2.cvtColor(center, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    warped_list = []
    mask_list = []

    warped_dir = d / 'warped'
    mask_dir = d / 'masks'
    has_masks = mask_dir.exists()

    for f in sorted(os.listdir(warped_dir)):
        if f.startswith(f'warped_{center_idx:06d}_from_'):
            w = cv2.imread(str(warped_dir / f))
            w = cv2.cvtColor(w, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            warped_list.append(w)

            neighbor = int(f.replace('.png', '').split('_')[3])
            if has_masks:
                m = np.load(str(mask_dir / f'mask_{center_idx:06d}_from_{neighbor:06d}.npy'))
            else:
                m = np.ones(center.shape[:2], dtype=bool)
            mask_list.append(m)

    return center, warped_list, mask_list


def compute_psnr(a, b):
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return 100.0
    return 10.0 * np.log10(1.0 / mse)


def compute_sharpness(img):
    gray = (np.mean(img, axis=2) * 255).clip(0, 255).astype(np.uint8)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def main():
    # ============================================================
    print("=" * 70)
    print("ANALYSIS 1: OCCLUSION MASK COVERAGE (things only)")
    print("=" * 70)

    d = base / 'things' / 'masks'
    coverages = []
    for f in sorted(os.listdir(d)):
        m = np.load(str(d / f))
        coverages.append(m.mean() * 100)

    coverages = np.array(coverages)
    print(f"Mean valid pixels: {coverages.mean():.1f}%")
    print(f"Min: {coverages.min():.1f}%, Max: {coverages.max():.1f}%, Std: {coverages.std():.1f}%")
    print("(sintel has no masks - was run without occlusion detection)")

    # ============================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 2: ALIGNMENT QUALITY (PSNR warped vs center)")
    print("=" * 70)

    for model in ['things', 'sintel']:
        print(f"\n{model.upper()}:")

        for center_idx in [10, 15, 20]:
            center, warped_list, mask_list = load_center_data(model, center_idx)

            psnrs = []
            for w in warped_list:
                psnrs.append(compute_psnr(center, w))

            print(f"  Frame {center_idx}: {len(warped_list)} neighbors, "
                  f"PSNR range [{min(psnrs):.1f}, {max(psnrs):.1f}] dB, "
                  f"mean {np.mean(psnrs):.1f} dB")

    # ============================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 3: FLOW MAGNITUDE STATISTICS")
    print("=" * 70)

    for model in ['things', 'sintel']:
        d = base / model / 'flows'
        files = sorted(os.listdir(d))
        mags_mean = []
        mags_p95 = []
        for f in files[:30]:
            try:
                flow = np.load(str(d / f)).astype(np.float32)
                mag = np.sqrt(flow[0]**2 + flow[1]**2)
                mags_mean.append(mag.mean())
                mags_p95.append(np.percentile(mag, 95))
            except (EOFError, ValueError):
                print(f"  WARNING: corrupt flow file: {f}")
                continue

        print(f"\n{model.upper()} (sample of {len(mags_mean)} flows):")
        print(f"  Mean magnitude: {np.mean(mags_mean):.1f} px")
        print(f"  95th percentile: {np.mean(mags_p95):.1f} px")

    # ============================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 4: DENOISING BY TEMPORAL AVERAGING")
    print("=" * 70)

    for model in ['things', 'sintel']:
        print(f"\n{model.upper()}:")

        for center_idx in [10, 15, 20]:
            center, warped_list, mask_list = load_center_data(model, center_idx)
            n = len(warped_list)

            all_frames = np.stack([center] + warped_list, axis=0)

            # Simple mean
            mean_result = np.mean(all_frames, axis=0)

            # Simple median
            median_result = np.median(all_frames, axis=0)

            # Masked weighted average
            weights = np.ones((n + 1, *center.shape[:2]), dtype=np.float32)
            for i, m in enumerate(mask_list):
                weights[i + 1] = m.astype(np.float32)
            weights_3d = weights[..., np.newaxis]
            weighted_result = np.sum(all_frames * weights_3d, axis=0) / np.sum(weights_3d, axis=0).clip(1e-6)

            # Temporal noise estimation
            temporal_std = np.std(all_frames, axis=0)
            mean_temporal_noise = temporal_std.mean()

            # Sharpness comparison
            sharp_orig = compute_sharpness(center)
            sharp_mean = compute_sharpness(mean_result)
            sharp_median = compute_sharpness(median_result)
            sharp_weighted = compute_sharpness(weighted_result)

            print(f"  Frame {center_idx} ({n} neighbors):")
            print(f"    Temporal noise (std): {mean_temporal_noise:.4f} ({mean_temporal_noise*255:.1f}/255)")
            print(f"    Sharpness: orig={sharp_orig:.0f}, mean={sharp_mean:.0f}, "
                  f"median={sharp_median:.0f}, weighted={sharp_weighted:.0f}")

            psnr_mean = compute_psnr(center, mean_result)
            psnr_median = compute_psnr(center, median_result)
            print(f"    PSNR vs orig: mean={psnr_mean:.1f} dB, median={psnr_median:.1f} dB")

    # ============================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 5: PIXEL-LEVEL TEMPORAL VARIANCE")
    print("=" * 70)

    for model in ['things', 'sintel']:
        center, warped_list, mask_list = load_center_data(model, 15)
        all_frames = np.stack([center] + warped_list, axis=0)

        temporal_var = np.var(all_frames, axis=0)
        var_gray = temporal_var.mean(axis=2)

        print(f"\n{model.upper()} frame 15 temporal variance:")
        for p in [25, 50, 75, 90, 95, 99]:
            v = np.percentile(var_gray, p)
            print(f"  p{p}: {v:.6f} (std={np.sqrt(v)*255:.1f}/255)")

        low_var = (var_gray < 0.001).mean() * 100
        med_var = ((var_gray >= 0.001) & (var_gray < 0.01)).mean() * 100
        high_var = (var_gray >= 0.01).mean() * 100
        print(f"  Low variance (<0.001): {low_var:.1f}%")
        print(f"  Medium variance: {med_var:.1f}%")
        print(f"  High variance (>0.01): {high_var:.1f}%")

    # ============================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 6: COMBINATION STRATEGIES COMPARISON")
    print("=" * 70)

    for model in ['things', 'sintel']:
        print(f"\n{model.upper()}:")

        sharpness_results = {k: [] for k in ['original', 'mean', 'median', 'weighted', 'wiener', 'trimmed_mean']}

        for center_idx in range(4, 26):
            center, warped_list, mask_list = load_center_data(model, center_idx)
            if len(warped_list) < 8:
                continue

            all_frames = np.stack([center] + warped_list, axis=0)
            n = all_frames.shape[0]

            # 1. Simple mean
            mean_result = np.mean(all_frames, axis=0)

            # 2. Median
            median_result = np.median(all_frames, axis=0)

            # 3. Masked weighted average
            weights = np.ones((n, *center.shape[:2]), dtype=np.float32)
            for i, m in enumerate(mask_list):
                weights[i + 1] = m.astype(np.float32)
            weights_3d = weights[..., np.newaxis]
            weighted_result = np.sum(all_frames * weights_3d, axis=0) / np.sum(weights_3d, axis=0).clip(1e-6)

            # 4. Wiener-like
            temporal_var = np.var(all_frames, axis=0).mean(axis=2, keepdims=True)
            noise_est = 0.001
            wiener_weight = noise_est / (temporal_var + noise_est)
            wiener_result = center * (1 - wiener_weight) + mean_result * wiener_weight

            # 5. Trimmed mean
            sorted_frames = np.sort(all_frames, axis=0)
            trimmed = sorted_frames[1:-1]
            trimmed_mean = np.mean(trimmed, axis=0)

            sharpness_results['original'].append(compute_sharpness(center))
            sharpness_results['mean'].append(compute_sharpness(mean_result))
            sharpness_results['median'].append(compute_sharpness(median_result))
            sharpness_results['weighted'].append(compute_sharpness(weighted_result))
            sharpness_results['wiener'].append(compute_sharpness(wiener_result))
            sharpness_results['trimmed_mean'].append(compute_sharpness(trimmed_mean))

        n_frames = len(sharpness_results['original'])
        print(f"  Across {n_frames} frames with 8+ neighbors:")
        for method, vals in sharpness_results.items():
            v = np.mean(vals)
            ratio = v / np.mean(sharpness_results['original']) * 100
            print(f"    {method:15s}: sharpness={v:8.1f} ({ratio:5.1f}% of original)")

    # ============================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 7: EDGE-SPECIFIC ALIGNMENT QUALITY")
    print("=" * 70)

    # Check if edges are preserved or blurred
    for model in ['things', 'sintel']:
        center, warped_list, mask_list = load_center_data(model, 15)

        # Detect edges in center frame
        gray_center = (np.mean(center, axis=2) * 255).astype(np.uint8)
        edges = cv2.Canny(gray_center, 50, 150) > 0  # boolean edge mask

        all_frames = np.stack([center] + warped_list, axis=0)
        mean_result = np.mean(all_frames, axis=0)

        # Compare temporal std at edges vs flat regions
        temporal_std = np.std(all_frames, axis=0).mean(axis=2)

        edge_std = temporal_std[edges].mean()
        flat_std = temporal_std[~edges].mean()

        # Sharpness at edges
        center_edge_vals = gray_center.astype(float)
        mean_gray = (np.mean(mean_result, axis=2) * 255).astype(np.uint8).astype(float)

        # Edge gradient magnitude
        center_grad = cv2.Sobel(gray_center, cv2.CV_64F, 1, 0)**2 + cv2.Sobel(gray_center, cv2.CV_64F, 0, 1)**2
        mean_u8 = (np.mean(mean_result, axis=2) * 255).astype(np.uint8)
        mean_grad = cv2.Sobel(mean_u8, cv2.CV_64F, 1, 0)**2 + cv2.Sobel(mean_u8, cv2.CV_64F, 0, 1)**2

        edge_sharpness_ratio = np.sqrt(mean_grad[edges]).mean() / np.sqrt(center_grad[edges]).mean()

        print(f"\n{model.upper()} frame 15:")
        print(f"  Edge pixels: {edges.sum()} ({edges.mean()*100:.1f}% of frame)")
        print(f"  Temporal std at edges: {edge_std:.4f} ({edge_std*255:.1f}/255)")
        print(f"  Temporal std at flat:  {flat_std:.4f} ({flat_std*255:.1f}/255)")
        print(f"  Edge/flat std ratio: {edge_std/flat_std:.1f}x")
        print(f"  Edge gradient preservation: {edge_sharpness_ratio*100:.1f}% of original")

    # ============================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 8: THINGS vs SINTEL DIRECT COMPARISON")
    print("=" * 70)

    for center_idx in [10, 15, 20]:
        print(f"\nFrame {center_idx}:")
        for model in ['things', 'sintel']:
            center, warped_list, mask_list = load_center_data(model, center_idx)
            all_frames = np.stack([center] + warped_list, axis=0)

            mean_result = np.mean(all_frames, axis=0)
            median_result = np.median(all_frames, axis=0)

            temporal_std = np.std(all_frames, axis=0).mean()
            sharp_orig = compute_sharpness(center)
            sharp_mean = compute_sharpness(mean_result)
            sharp_median = compute_sharpness(median_result)

            print(f"  {model:8s}: temporal_std={temporal_std:.4f}, "
                  f"sharpness orig/mean/median = {sharp_orig:.0f}/{sharp_mean:.0f}/{sharp_median:.0f}, "
                  f"mean_retain={sharp_mean/sharp_orig*100:.0f}%")

    # ============================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 9: SAVE SAMPLE OUTPUTS FOR VISUAL INSPECTION")
    print("=" * 70)

    out_dir = Path('data/archive/raft_modal_extract/analysis')
    out_dir.mkdir(exist_ok=True)

    for model in ['things', 'sintel']:
        center, warped_list, mask_list = load_center_data(model, 15)
        all_frames = np.stack([center] + warped_list, axis=0)

        results = {
            'original': center,
            'mean': np.mean(all_frames, axis=0),
            'median': np.median(all_frames, axis=0),
        }

        # Trimmed mean
        sorted_frames = np.sort(all_frames, axis=0)
        results['trimmed_mean'] = np.mean(sorted_frames[1:-1], axis=0)

        # Weighted (things only has masks)
        n = all_frames.shape[0]
        weights = np.ones((n, *center.shape[:2]), dtype=np.float32)
        for i, m in enumerate(mask_list):
            weights[i + 1] = m.astype(np.float32)
        weights_3d = weights[..., np.newaxis]
        results['weighted'] = np.sum(all_frames * weights_3d, axis=0) / np.sum(weights_3d, axis=0).clip(1e-6)

        # Wiener
        temporal_var = np.var(all_frames, axis=0).mean(axis=2, keepdims=True)
        noise_est = 0.001
        wiener_weight = noise_est / (temporal_var + noise_est)
        results['wiener'] = center * (1 - wiener_weight) + results['mean'] * wiener_weight

        # Save full frames
        for name, img in results.items():
            img_bgr = cv2.cvtColor((img.clip(0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_dir / f'{model}_frame15_{name}.png'), img_bgr)

        # Save 256x256 crops from an interesting region (center of frame)
        cy, cx = center.shape[0] // 2, center.shape[1] // 2
        for name, img in results.items():
            crop = img[cy-128:cy+128, cx-128:cx+128]
            crop_bgr = cv2.cvtColor((crop.clip(0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_dir / f'{model}_frame15_{name}_crop.png'), crop_bgr)

        print(f"  Saved {model} frame 15: {len(results)} methods x (full + crop)")

    # Also save difference maps
    for model in ['things', 'sintel']:
        center, warped_list, mask_list = load_center_data(model, 15)
        all_frames = np.stack([center] + warped_list, axis=0)
        mean_result = np.mean(all_frames, axis=0)

        # Amplified difference (5x)
        diff = np.abs(center - mean_result) * 5
        diff_bgr = cv2.cvtColor((diff.clip(0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_dir / f'{model}_frame15_diff_5x.png'), diff_bgr)

        # Temporal std map (normalized to visible range)
        temporal_std = np.std(all_frames, axis=0)
        std_vis = (temporal_std * 10).clip(0, 1)  # 10x amplification
        std_bgr = cv2.cvtColor((std_vis * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_dir / f'{model}_frame15_temporal_std_10x.png'), std_bgr)

    print("  Saved difference and temporal std maps")
    print(f"\nAll outputs saved to: {out_dir}")
    print("\nDone!")


if __name__ == '__main__':
    main()
