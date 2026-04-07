"""Compare denoising approaches: DRUNet teacher vs temporal methods vs DINO.

Generates side-by-side comparison crops for visual evaluation.
"""
import os
os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"

import sys
import gc
import numpy as np
from pathlib import Path
import cv2
import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from lib.paths import add_kair_to_path

BASE = ROOT / "data" / "archive" / "raft_modal_extract"
IMG_DIR = BASE / "things" / "originals"
WARPED_DIR = BASE / "things" / "warped"
MASK_DIR = BASE / "things" / "masks"
FEAT_DIR = BASE / "dino_features"
OUT_DIR = BASE / "comparison"
OUT_DIR.mkdir(exist_ok=True)

PATCH_SIZE = 16


def load_frame_rgb(idx):
    img = cv2.imread(str(IMG_DIR / f"original_{idx:06d}.png"))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def load_warped_frames(center_idx):
    """Load RAFT-warped neighbors and masks for a center frame."""
    warped_list = []
    mask_list = []
    for f in sorted(WARPED_DIR.glob(f"warped_{center_idx:06d}_from_*.png")):
        w = cv2.imread(str(f))
        w = cv2.cvtColor(w, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        warped_list.append(w)
        neighbor = int(f.stem.split("_")[3])
        mf = MASK_DIR / f"mask_{center_idx:06d}_from_{neighbor:06d}.npy"
        if mf.exists():
            mask_list.append(np.load(str(mf)))
        else:
            mask_list.append(np.ones(w.shape[:2], dtype=bool))
    return warped_list, mask_list


def l2_normalize(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def sharpness(img):
    gray = (np.mean(img, axis=2) * 255).clip(0, 255).astype(np.uint8)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def noise_est(img):
    gray = (np.mean(img, axis=2) * 255).clip(0, 255).astype(np.uint8)
    return np.median(np.abs(cv2.Laplacian(gray, cv2.CV_64F))) / 0.6745


# ================================================================
# METHOD 1: DRUNet Teacher inference
# ================================================================
def run_drunet_teacher(img_rgb):
    """Run DRUNet teacher on a single frame. Returns denoised RGB float32."""
    add_kair_to_path()
    from models.network_unet import UNetRes

    ckpt_path = ROOT / "checkpoints" / "drunet_teacher" / "final.pth"
    device = "cuda"

    # Build model
    model = UNetRes(in_nc=3, out_nc=3, nc=[64, 128, 256, 512], nb=4, act_mode="R", bias=False)
    state = torch.load(str(ckpt_path), map_location="cpu")
    if "params" in state:
        state = state["params"]
    model.load_state_dict(state, strict=True)
    model = model.to(device).half().eval()

    n_params = sum(p.numel() for p in model.parameters())
    vram = torch.cuda.memory_allocated() / 1024**2
    print(f"  DRUNet teacher: {n_params/1e6:.1f}M params, {vram:.0f}MB VRAM")

    # Inference
    tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0).to(device).half()
    # Pad to multiple of 8
    _, _, h, w = tensor.shape
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    if pad_h or pad_w:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")

    with torch.no_grad():
        out = model(tensor)

    out = out[:, :, :h, :w]
    result = out[0].float().cpu().numpy().transpose(1, 2, 0).clip(0, 1)

    del model, tensor, out
    torch.cuda.empty_cache()
    gc.collect()

    return result


# ================================================================
# METHOD 2: RAFT temporal median
# ================================================================
def raft_temporal_median(center_img, warped_list, mask_list):
    all_frames = np.stack([center_img] + warped_list, axis=0)
    return np.median(all_frames, axis=0).astype(np.float32)


# ================================================================
# METHOD 3: RAFT masked weighted average
# ================================================================
def raft_masked_average(center_img, warped_list, mask_list):
    all_frames = np.stack([center_img] + warped_list, axis=0)
    n = all_frames.shape[0]
    H, W = center_img.shape[:2]
    weights = np.ones((n, H, W), dtype=np.float32)
    weights[0] = 1.0
    for i, m in enumerate(mask_list):
        weights[i + 1] = m.astype(np.float32)
    weights_3d = weights[..., np.newaxis]
    return (np.sum(all_frames * weights_3d, axis=0) / np.sum(weights_3d, axis=0).clip(1e-6)).astype(np.float32)


# ================================================================
# METHOD 4: DINO feature-matched average
# ================================================================
def dino_matched_average(center_img, center_idx, neighbor_indices, sim_threshold=0.5):
    """Match patches by DINO features, average pixels of matched patches."""
    H, W = center_img.shape[:2]

    # Load center features
    center_data = np.load(str(FEAT_DIR / f"original_{center_idx:06d}.npz"))
    center_patches = center_data["patches"][0]
    grid_h = int(center_data["grid_h"])
    grid_w = int(center_data["grid_w"])
    center_norm = l2_normalize(center_patches)

    accumulated = center_img.copy()
    weight_map = np.ones((H, W, 1), dtype=np.float32)

    for ni in neighbor_indices:
        neigh_img = load_frame_rgb(ni)
        neigh_data = np.load(str(FEAT_DIR / f"original_{ni:06d}.npz"))
        neigh_patches = neigh_data["patches"][0]
        neigh_norm = l2_normalize(neigh_patches)

        sim_matrix = center_norm @ neigh_norm.T
        best_idx = sim_matrix.argmax(axis=1)
        best_sim = sim_matrix.max(axis=1)

        for pi in range(len(best_idx)):
            cy = (pi // grid_w) * PATCH_SIZE
            cx = (pi % grid_w) * PATCH_SIZE
            if cy + PATCH_SIZE > H or cx + PATCH_SIZE > W:
                continue
            mi = best_idx[pi]
            my = (mi // grid_w) * PATCH_SIZE
            mx = (mi % grid_w) * PATCH_SIZE
            if my + PATCH_SIZE > neigh_img.shape[0] or mx + PATCH_SIZE > neigh_img.shape[1]:
                continue
            sim = best_sim[pi]
            if sim > sim_threshold:
                w = sim
                accumulated[cy:cy+PATCH_SIZE, cx:cx+PATCH_SIZE] += neigh_img[my:my+PATCH_SIZE, mx:mx+PATCH_SIZE] * w
                weight_map[cy:cy+PATCH_SIZE, cx:cx+PATCH_SIZE] += w

    return (accumulated / weight_map.clip(1e-6)).astype(np.float32)


# ================================================================
# METHOD 5: Hybrid DINO + RAFT
# Use DINO to decide WHICH patches to average, RAFT for sub-pixel alignment
# ================================================================
def hybrid_dino_raft(center_img, center_idx, warped_list, mask_list, neighbor_indices):
    """
    For each patch:
    - Compute DINO similarity to decide if this patch is a good match
    - If good DINO match AND RAFT mask says valid: use RAFT-warped pixels (sub-pixel aligned)
    - If good DINO match but no RAFT mask: use DINO-matched patch (block-level)
    - If bad DINO match: skip this neighbor for this patch
    """
    H, W = center_img.shape[:2]

    center_data = np.load(str(FEAT_DIR / f"original_{center_idx:06d}.npz"))
    center_patches = center_data["patches"][0]
    grid_h = int(center_data["grid_h"])
    grid_w = int(center_data["grid_w"])
    center_norm = l2_normalize(center_patches)

    accumulated = center_img.copy()
    weight_map = np.ones((H, W, 1), dtype=np.float32)

    for wi, ni in enumerate(neighbor_indices):
        if wi >= len(warped_list):
            break
        warped_img = warped_list[wi]
        mask = mask_list[wi]

        neigh_data = np.load(str(FEAT_DIR / f"original_{ni:06d}.npz"))
        neigh_norm = l2_normalize(neigh_data["patches"][0])

        # Per-patch DINO similarity (same spatial position)
        sim_same_pos = (center_norm * neigh_norm).sum(axis=1)  # (N,)

        for pi in range(len(sim_same_pos)):
            cy = (pi // grid_w) * PATCH_SIZE
            cx = (pi % grid_w) * PATCH_SIZE
            if cy + PATCH_SIZE > H or cx + PATCH_SIZE > W:
                continue

            sim = sim_same_pos[pi]
            if sim < 0.7:
                continue  # bad semantic match, skip entirely

            # Check RAFT mask coverage in this patch
            patch_mask = mask[cy:cy+PATCH_SIZE, cx:cx+PATCH_SIZE]
            mask_coverage = patch_mask.mean()

            if mask_coverage > 0.8:
                # Good RAFT alignment: use sub-pixel warped pixels
                w = sim * mask_coverage
                accumulated[cy:cy+PATCH_SIZE, cx:cx+PATCH_SIZE] += warped_img[cy:cy+PATCH_SIZE, cx:cx+PATCH_SIZE] * w
                weight_map[cy:cy+PATCH_SIZE, cx:cx+PATCH_SIZE] += w
            elif sim > 0.9:
                # Great DINO match but RAFT failed (occlusion): still use warped with lower weight
                w = sim * 0.3
                accumulated[cy:cy+PATCH_SIZE, cx:cx+PATCH_SIZE] += warped_img[cy:cy+PATCH_SIZE, cx:cx+PATCH_SIZE] * w
                weight_map[cy:cy+PATCH_SIZE, cx:cx+PATCH_SIZE] += w

    return (accumulated / weight_map.clip(1e-6)).astype(np.float32)


# ================================================================
# METHOD 6: DINO-matched temporal median
# ================================================================
def dino_matched_median(center_img, center_idx, neighbor_indices, sim_threshold=0.7):
    """For each patch, collect DINO-matched pixels and take median."""
    H, W = center_img.shape[:2]

    center_data = np.load(str(FEAT_DIR / f"original_{center_idx:06d}.npz"))
    center_patches = center_data["patches"][0]
    grid_h = int(center_data["grid_h"])
    grid_w = int(center_data["grid_w"])
    center_norm = l2_normalize(center_patches)

    # Collect all matched patches per position
    # Pre-allocate: max 1 center + N neighbors
    max_frames = 1 + len(neighbor_indices)
    patch_stack = np.zeros((max_frames, H, W, 3), dtype=np.float32)
    patch_count = np.ones((H, W), dtype=np.int32)  # center always included
    patch_stack[0] = center_img

    for ni_idx, ni in enumerate(neighbor_indices):
        neigh_img = load_frame_rgb(ni)
        neigh_data = np.load(str(FEAT_DIR / f"original_{ni:06d}.npz"))
        neigh_norm = l2_normalize(neigh_data["patches"][0])

        sim_matrix = center_norm @ neigh_norm.T
        best_idx = sim_matrix.argmax(axis=1)
        best_sim = sim_matrix.max(axis=1)

        for pi in range(len(best_idx)):
            if best_sim[pi] < sim_threshold:
                continue
            cy = (pi // grid_w) * PATCH_SIZE
            cx = (pi % grid_w) * PATCH_SIZE
            if cy + PATCH_SIZE > H or cx + PATCH_SIZE > W:
                continue
            mi = best_idx[pi]
            my = (mi // grid_w) * PATCH_SIZE
            mx = (mi % grid_w) * PATCH_SIZE
            if my + PATCH_SIZE > neigh_img.shape[0] or mx + PATCH_SIZE > neigh_img.shape[1]:
                continue

            # Add to stack
            slot = patch_count[cy, cx]
            if slot < max_frames:
                patch_stack[slot, cy:cy+PATCH_SIZE, cx:cx+PATCH_SIZE] = neigh_img[my:my+PATCH_SIZE, mx:mx+PATCH_SIZE]
                patch_count[cy:cy+PATCH_SIZE, cx:cx+PATCH_SIZE] = slot + 1

    # Take median per pixel using only populated slots
    result = np.zeros_like(center_img)
    max_count = patch_count.max()
    for count_val in range(1, max_count + 1):
        mask = patch_count == count_val
        if not mask.any():
            continue
        # For pixels with this count, take median of first count_val frames
        for c in range(3):
            vals = patch_stack[:count_val, :, :, c]
            med = np.median(vals, axis=0)
            result[:, :, c] = np.where(mask, med, result[:, :, c])

    return result.astype(np.float32)


def save_crop(img_rgb, path, cy, cx, sz):
    crop = img_rgb[cy:cy+sz, cx:cx+sz]
    crop_bgr = cv2.cvtColor((crop.clip(0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), crop_bgr)


def save_full(img_rgb, path):
    img_bgr = cv2.cvtColor((img_rgb.clip(0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), img_bgr)


def make_comparison_strip(images_dict, cy, cx, sz, path):
    """Create a labeled horizontal strip of crops."""
    crops = []
    for name, img in images_dict.items():
        crop = img[cy:cy+sz, cx:cx+sz]
        crop_u8 = (crop.clip(0, 1) * 255).astype(np.uint8)
        # Add label
        crop_bgr = cv2.cvtColor(crop_u8, cv2.COLOR_RGB2BGR)
        cv2.putText(crop_bgr, name, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(crop_bgr, name, (4, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        crops.append(crop_bgr)
    strip = np.concatenate(crops, axis=1)
    cv2.imwrite(str(path), strip)


def main():
    CENTER = 15
    NEIGHBORS = [11, 12, 13, 14, 16, 17, 18, 19]

    print("Loading center frame and RAFT data...")
    center_img = load_frame_rgb(CENTER)
    warped_list, mask_list = load_warped_frames(CENTER)
    H, W = center_img.shape[:2]
    print(f"  Frame {CENTER}: {W}x{H}, {len(warped_list)} warped neighbors")

    results = {}
    results["original"] = center_img

    # Method 1: DRUNet teacher
    print("\nRunning DRUNet teacher...")
    results["drunet_teacher"] = run_drunet_teacher(center_img)

    # Method 2: RAFT median
    print("Computing RAFT median...")
    results["raft_median"] = raft_temporal_median(center_img, warped_list, mask_list)

    # Method 3: RAFT masked average
    print("Computing RAFT masked average...")
    results["raft_weighted"] = raft_masked_average(center_img, warped_list, mask_list)

    # Method 4: DINO matched average
    print("Computing DINO matched average...")
    results["dino_avg"] = dino_matched_average(center_img, CENTER, NEIGHBORS, sim_threshold=0.5)

    # Method 5: Hybrid DINO + RAFT
    print("Computing hybrid DINO+RAFT...")
    results["hybrid"] = hybrid_dino_raft(center_img, CENTER, warped_list, mask_list, NEIGHBORS)

    # Method 6: DINO matched median
    print("Computing DINO matched median...")
    results["dino_median"] = dino_matched_median(center_img, CENTER, NEIGHBORS, sim_threshold=0.7)

    # ============================================================
    # Metrics
    # ============================================================
    print("\n" + "=" * 80)
    print(f"{'Method':<20} {'Sharpness':>10} {'Noise':>8} {'SNR':>8} {'Sharp%':>8} {'Noise%':>8}")
    print("-" * 80)
    s_orig = sharpness(center_img)
    n_orig = noise_est(center_img)
    for name, img in results.items():
        s = sharpness(img)
        n = noise_est(img)
        snr = s / n if n > 0 else 0
        sp = s / s_orig * 100
        np_ = n / n_orig * 100
        print(f"{name:<20} {s:>10.1f} {n:>8.2f} {snr:>8.2f} {sp:>7.0f}% {np_:>7.0f}%")

    # ============================================================
    # Save comparison images
    # ============================================================
    print("\nSaving comparison images...")

    # Find interesting crops
    gray = (np.mean(center_img, axis=2) * 255).astype(np.uint8)
    edges = cv2.Canny(gray, 30, 100)

    crops = []
    # Crop 1: brightest area with detail
    best_score = 0
    best_pos = (0, 0)
    for y in range(0, H - 384, 32):
        for x in range(0, W - 384, 32):
            brightness = gray[y:y+384, x:x+384].mean()
            edge_density = edges[y:y+384, x:x+384].mean()
            score = edge_density * min(brightness, 80)
            if score > best_score:
                best_score = score
                best_pos = (y, x)
    crops.append(("detail", best_pos[0], best_pos[1], 384))

    # Crop 2: center of frame
    crops.append(("center", H // 2 - 192, W // 2 - 192, 384))

    # Crop 3: brightest region
    best_bright = 0
    best_bp = (0, 0)
    for y in range(0, H - 384, 64):
        for x in range(0, W - 384, 64):
            b = gray[y:y+384, x:x+384].mean()
            if b > best_bright:
                best_bright = b
                best_bp = (y, x)
    crops.append(("bright", best_bp[0], best_bp[1], 384))

    # Save individual crops and comparison strips
    for crop_name, cy, cx, sz in crops:
        # Save comparison strip with key methods
        key_results = {
            "original": results["original"],
            "drunet": results["drunet_teacher"],
            "raft_med": results["raft_median"],
            "dino_avg": results["dino_avg"],
            "hybrid": results["hybrid"],
            "dino_med": results["dino_median"],
        }
        make_comparison_strip(key_results, cy, cx, sz, OUT_DIR / f"strip_{crop_name}.png")

        # Save individual crops
        for name, img in results.items():
            save_crop(img, OUT_DIR / f"crop_{crop_name}_{name}.png", cy, cx, sz)

    # Save full frames for key methods
    for name in ["original", "drunet_teacher", "raft_median", "dino_avg", "hybrid", "dino_median"]:
        save_full(results[name], OUT_DIR / f"full_{name}.png")

    print(f"Saved to: {OUT_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
