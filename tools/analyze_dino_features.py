"""Analyze DINOv3 features for cross-frame denoising potential."""
import os
os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"

import numpy as np
from pathlib import Path
import cv2
from sklearn.decomposition import PCA

feat_dir = Path("data/archive/raft_modal_extract/dino_features")
img_dir = Path("data/archive/raft_modal_extract/things/originals")
out_dir = Path("data/archive/raft_modal_extract/analysis")
out_dir.mkdir(exist_ok=True)

PATCH_SIZE = 16


def load_frame(idx):
    img = cv2.imread(str(img_dir / f"original_{idx:06d}.png"))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def sharpness(img):
    gray = (np.mean(img, axis=2) * 255).clip(0, 255).astype(np.uint8)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def noise_estimate(img):
    gray = (np.mean(img, axis=2) * 255).clip(0, 255).astype(np.uint8)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return np.median(np.abs(lap)) / 0.6745


def l2_normalize(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def main():
    # Load all features
    print("Loading features...")
    all_patches = {}
    all_cls = {}
    grid_h = grid_w = 0
    for f in sorted(feat_dir.glob("*.npz")):
        idx = int(f.stem.split("_")[1])
        data = np.load(str(f))
        all_patches[idx] = data["patches"][0]  # (N, 384)
        all_cls[idx] = data["cls"][0]  # (384,)
        grid_h = int(data["grid_h"])
        grid_w = int(data["grid_w"])

    print(f"Loaded {len(all_patches)} frames, grid {grid_h}x{grid_w}, {all_patches[0].shape[1]}-dim")

    # ============================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 1: FEATURE SIMILARITY ACROSS FRAMES")
    print("=" * 70)

    for dist in [1, 2, 4, 8]:
        sims = []
        for i in range(dist, 30):
            p1_n = l2_normalize(all_patches[i])
            p2_n = l2_normalize(all_patches[i - dist])
            cos_sim = (p1_n * p2_n).sum(axis=1)
            sims.append(cos_sim.mean())
        print(f"  Distance {dist}: mean cosine sim = {np.mean(sims):.4f} (same position)")

    print("\nCLS token (global) similarity:")
    for dist in [1, 2, 4, 8]:
        sims = []
        for i in range(dist, 30):
            c1 = all_cls[i] / (np.linalg.norm(all_cls[i]) + 1e-8)
            c2 = all_cls[i - dist] / (np.linalg.norm(all_cls[i - dist]) + 1e-8)
            sims.append(np.dot(c1, c2))
        print(f"  Distance {dist}: mean cosine sim = {np.mean(sims):.4f}")

    # ============================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 2: CROSS-FRAME PATCH MATCHING")
    print("=" * 70)

    center_idx = 15
    center_norm = l2_normalize(all_patches[center_idx])

    for neighbor_idx in [14, 13, 11, 7]:
        neigh_norm = l2_normalize(all_patches[neighbor_idx])
        sim_matrix = center_norm @ neigh_norm.T
        best_idx = sim_matrix.argmax(axis=1)
        best_sim = sim_matrix.max(axis=1)

        same_pos = (best_idx == np.arange(len(best_idx))).mean() * 100
        match_row = best_idx // grid_w
        match_col = best_idx % grid_w
        orig_row = np.arange(len(best_idx)) // grid_w
        orig_col = np.arange(len(best_idx)) % grid_w
        dist_patches = np.sqrt((match_row - orig_row) ** 2 + (match_col - orig_col) ** 2)
        within_1 = (dist_patches <= 1).mean() * 100
        within_3 = (dist_patches <= 3).mean() * 100

        d = abs(center_idx - neighbor_idx)
        print(
            f"  Frame {neighbor_idx} (dist={d}): "
            f"sim={best_sim.mean():.4f}, "
            f"same_pos={same_pos:.0f}%, "
            f"within_1={within_1:.0f}%, "
            f"within_3={within_3:.0f}%, "
            f"mean_disp={dist_patches.mean():.1f}"
        )

    # ============================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 3: PCA VISUALIZATION")
    print("=" * 70)

    patches = all_patches[15]
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(patches)
    print(f"PCA explained variance: {pca.explained_variance_ratio_[:3] * 100}")

    pca_map = pca_features.reshape(grid_h, grid_w, 3)
    for c in range(3):
        mn, mx = pca_map[:, :, c].min(), pca_map[:, :, c].max()
        pca_map[:, :, c] = (pca_map[:, :, c] - mn) / (mx - mn + 1e-8)
    pca_img = (pca_map * 255).astype(np.uint8)
    pca_large = cv2.resize(pca_img, (grid_w * 8, grid_h * 8), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(out_dir / "dino_pca_frame15.png"), cv2.cvtColor(pca_large, cv2.COLOR_RGB2BGR))
    print("Saved PCA visualization")

    # Multi-frame PCA (same PCA basis across frames 13-17)
    all_feats_multi = np.concatenate([all_patches[i] for i in [13, 14, 15, 16, 17]], axis=0)
    pca_multi = PCA(n_components=3)
    pca_multi_feats = pca_multi.fit_transform(all_feats_multi)

    for fi, frame_idx in enumerate([13, 14, 15, 16, 17]):
        start = fi * grid_h * grid_w
        end = start + grid_h * grid_w
        frame_pca = pca_multi_feats[start:end].reshape(grid_h, grid_w, 3)
        for c in range(3):
            vmin, vmax = pca_multi_feats[:, c].min(), pca_multi_feats[:, c].max()
            frame_pca[:, :, c] = (frame_pca[:, :, c] - vmin) / (vmax - vmin + 1e-8)
        frame_img = (frame_pca * 255).astype(np.uint8)
        frame_large = cv2.resize(frame_img, (grid_w * 8, grid_h * 8), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(
            str(out_dir / f"dino_pca_multi_frame{frame_idx}.png"),
            cv2.cvtColor(frame_large, cv2.COLOR_RGB2BGR),
        )
    print("Saved multi-frame PCA (consistent basis across frames 13-17)")

    # ============================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 4: FEATURE-GUIDED PIXEL AVERAGING (THE KEY TEST)")
    print("=" * 70)

    center_img = load_frame(15)
    H, W = center_img.shape[:2]
    center_norm = l2_normalize(all_patches[15])

    neighbor_indices = [11, 12, 13, 14, 16, 17, 18, 19]

    # Accumulate pixel data
    accumulated = center_img.copy()
    weight_map = np.ones((H, W, 1), dtype=np.float32)

    sim_threshold = 0.5
    n_good_matches = 0
    n_total_matches = 0

    for ni in neighbor_indices:
        neigh_img = load_frame(ni)
        neigh_norm = l2_normalize(all_patches[ni])

        sim_matrix = center_norm @ neigh_norm.T
        best_idx = sim_matrix.argmax(axis=1)
        best_sim = sim_matrix.max(axis=1)

        for patch_i in range(len(best_idx)):
            cy = (patch_i // grid_w) * PATCH_SIZE
            cx = (patch_i % grid_w) * PATCH_SIZE
            if cy + PATCH_SIZE > H or cx + PATCH_SIZE > W:
                continue

            mi = best_idx[patch_i]
            my = (mi // grid_w) * PATCH_SIZE
            mx = (mi % grid_w) * PATCH_SIZE
            if my + PATCH_SIZE > neigh_img.shape[0] or mx + PATCH_SIZE > neigh_img.shape[1]:
                continue

            n_total_matches += 1
            sim = best_sim[patch_i]
            if sim > sim_threshold:
                n_good_matches += 1
                w = sim
                accumulated[cy : cy + PATCH_SIZE, cx : cx + PATCH_SIZE] += (
                    neigh_img[my : my + PATCH_SIZE, mx : mx + PATCH_SIZE] * w
                )
                weight_map[cy : cy + PATCH_SIZE, cx : cx + PATCH_SIZE] += w

    result = accumulated / weight_map.clip(1e-6)

    s_orig = sharpness(center_img)
    s_dino = sharpness(result)
    n_orig = noise_estimate(center_img)
    n_dino = noise_estimate(result)

    print(f"Good matches: {n_good_matches}/{n_total_matches} ({n_good_matches/n_total_matches*100:.0f}%)")
    print(f"Avg contributing frames/pixel: {weight_map.mean():.1f}")
    print(f"Original:     sharpness={s_orig:.1f}, noise={n_orig:.2f}, SNR={s_orig/n_orig:.2f}")
    print(f"DINO-matched: sharpness={s_dino:.1f}, noise={n_dino:.2f}, SNR={s_dino/n_dino:.2f}")
    print(f"Sharpness retention: {s_dino/s_orig*100:.0f}%")
    print(f"Noise reduction: {n_dino/n_orig*100:.0f}%")
    print(f"SNR improvement: {(s_dino/n_dino)/(s_orig/n_orig):.2f}x")

    # Save results
    result_bgr = cv2.cvtColor((result.clip(0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_dir / "dino_matched_avg_frame15.png"), result_bgr)

    # Save crop comparisons
    cy, cx, sz = 768, 480, 384
    for name, img in [("original", center_img), ("dino_matched", result)]:
        crop = img[cy : cy + sz, cx : cx + sz]
        crop_bgr = cv2.cvtColor((crop.clip(0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_dir / f"dino_{name}_crop.png"), crop_bgr)

    # ============================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 5: COMPARISON WITH RAFT TEMPORAL AVERAGING")
    print("=" * 70)

    # Load RAFT warped frames for same center
    warped_dir = Path("data/archive/raft_modal_extract/things/warped")
    mask_dir = Path("data/archive/raft_modal_extract/things/masks")

    raft_frames = [center_img.copy()]
    for f in sorted(warped_dir.glob(f"warped_{15:06d}_from_*.png")):
        w = cv2.imread(str(f))
        w = cv2.cvtColor(w, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        raft_frames.append(w)

    raft_stack = np.stack(raft_frames, axis=0)
    raft_mean = np.mean(raft_stack, axis=0)
    raft_median = np.median(raft_stack, axis=0)

    methods = {
        "original": center_img,
        "raft_mean": raft_mean,
        "raft_median": raft_median,
        "dino_matched": result,
    }

    print(f"\n{'Method':<20} {'Sharpness':>10} {'Noise':>8} {'SNR':>8} {'Sharp%':>8} {'Noise%':>8}")
    print("-" * 70)
    for name, img in methods.items():
        s = sharpness(img)
        n = noise_estimate(img)
        snr = s / n
        sp = s / s_orig * 100
        np_ = n / n_orig * 100
        print(f"{name:<20} {s:>10.1f} {n:>8.2f} {snr:>8.2f} {sp:>7.0f}% {np_:>7.0f}%")

    print("\nDone! All outputs saved to:", out_dir)


if __name__ == "__main__":
    main()
