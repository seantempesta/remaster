"""Compare upscaling results: PSNR and SSIM against 1080p ground truth.
Also generates side-by-side comparison frames."""
import os, glob
import numpy as np
import cv2

data_dir = r'C:\Users\sean\src\upscale-experiment\data'
gt_dir = os.path.join(data_dir, 'frames_1080p_gt')
MAX_FRAMES = 150

methods = {
    'Real-ESRGAN (baseline)': os.path.join(data_dir, 'frames_realesrgan'),
    'Flow-Fused + Real-ESRGAN': os.path.join(data_dir, 'frames_warp_fuse_sr'),
    'Bicubic (sanity check)': None,  # we'll generate on the fly
}


def compute_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255.0 ** 2 / mse)


def compute_ssim(img1, img2):
    """Simple SSIM (per-channel, averaged)."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(np.mean(ssim_map))


def main():
    gt_frames = sorted(glob.glob(os.path.join(gt_dir, '*.png')))[:MAX_FRAMES]
    input_frames = sorted(glob.glob(os.path.join(data_dir, 'frames_480p', '*.png')))[:MAX_FRAMES]

    results = {}
    for method_name, method_dir in methods.items():
        psnrs, ssims = [], []
        for i, gt_path in enumerate(gt_frames):
            gt = cv2.imread(gt_path, cv2.IMREAD_COLOR)

            if method_dir is None:
                # Bicubic baseline
                lr = cv2.imread(input_frames[i], cv2.IMREAD_COLOR)
                sr = cv2.resize(lr, (1920, 1080), interpolation=cv2.INTER_CUBIC)
            else:
                sr_path = os.path.join(method_dir, f'frame_{i+1:05d}.png')
                if not os.path.exists(sr_path):
                    continue
                sr = cv2.imread(sr_path, cv2.IMREAD_COLOR)

            if sr.shape != gt.shape:
                sr = cv2.resize(sr, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LANCZOS4)

            psnrs.append(compute_psnr(gt, sr))
            ssims.append(compute_ssim(gt, sr))

        if psnrs:
            results[method_name] = {
                'psnr_mean': np.mean(psnrs),
                'psnr_std': np.std(psnrs),
                'ssim_mean': np.mean(ssims),
                'ssim_std': np.std(ssims),
                'n_frames': len(psnrs),
            }

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS: Video Upscaling Comparison (480p -> 1080p)")
    print("=" * 70)
    print(f"{'Method':<30} {'PSNR (dB)':>12} {'SSIM':>12} {'Frames':>8}")
    print("-" * 70)
    for method, r in sorted(results.items(), key=lambda x: x[1]['psnr_mean']):
        print(f"{method:<30} {r['psnr_mean']:>8.2f} +/- {r['psnr_std']:.2f} "
              f"{r['ssim_mean']:>6.4f} +/- {r['ssim_std']:.4f} {r['n_frames']:>6}")
    print("=" * 70)

    # Generate a few side-by-side comparison frames
    compare_dir = os.path.join(data_dir, 'comparisons')
    os.makedirs(compare_dir, exist_ok=True)
    sample_indices = [0, 30, 60, 90, 120]

    for idx in sample_indices:
        if idx >= len(gt_frames):
            continue
        gt = cv2.imread(gt_frames[idx], cv2.IMREAD_COLOR)
        lr = cv2.imread(input_frames[idx], cv2.IMREAD_COLOR)
        lr_up = cv2.resize(lr, (1920, 1080), interpolation=cv2.INTER_CUBIC)

        panels = [lr_up]
        labels = ['Bicubic']

        for method_name, method_dir in methods.items():
            if method_dir is None:
                continue
            sr_path = os.path.join(method_dir, f'frame_{idx+1:05d}.png')
            if os.path.exists(sr_path):
                sr = cv2.imread(sr_path, cv2.IMREAD_COLOR)
                if sr.shape != gt.shape:
                    sr = cv2.resize(sr, (gt.shape[1], gt.shape[0]))
                panels.append(sr)
                labels.append(method_name.split('(')[0].strip())

        panels.append(gt)
        labels.append('Ground Truth')

        # Add labels to panels
        for j, (panel, label) in enumerate(zip(panels, labels)):
            cv2.putText(panel, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        comparison = np.hstack(panels)
        cv2.imwrite(os.path.join(compare_dir, f'compare_frame_{idx:05d}.png'), comparison)

    print(f"\nSide-by-side comparisons saved to {compare_dir}")


if __name__ == '__main__':
    main()
