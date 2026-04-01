"""Phase 1 of Test A: Warp and fuse frames using RAFT flow (CPU only, no GPU needed)."""
import os, glob, time
import numpy as np
import cv2
import torch
import torch.nn.functional as F

data_dir = r'C:\Users\sean\src\upscale-experiment\data'
frames_dir = os.path.join(data_dir, 'frames_480p')
flow_dir = os.path.join(data_dir, 'flow_npy')
fused_dir = os.path.join(data_dir, 'frames_fused_480p')
os.makedirs(fused_dir, exist_ok=True)

MAX_FRAMES = 150


def warp_frame(frame_np, flow_np):
    H, W = frame_np.shape[:2]
    frame = torch.from_numpy(frame_np).float().permute(2, 0, 1).unsqueeze(0)
    flow = torch.from_numpy(flow_np).float()
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing='ij'
    )
    new_x = 2.0 * (grid_x + flow[..., 0]) / (W - 1) - 1.0
    new_y = 2.0 * (grid_y + flow[..., 1]) / (H - 1) - 1.0
    grid = torch.stack([new_x, new_y], dim=-1).unsqueeze(0)
    warped = F.grid_sample(frame, grid, mode='bilinear', padding_mode='border', align_corners=True)
    return warped.squeeze(0).permute(1, 2, 0).clamp(0, 255).byte().numpy()


def compute_occlusion_mask(flow_fwd_np, flow_bwd_np, threshold=1.5):
    H, W = flow_fwd_np.shape[:2]
    flow_bwd_t = torch.from_numpy(flow_bwd_np).float().permute(2, 0, 1).unsqueeze(0)
    flow_fwd = torch.from_numpy(flow_fwd_np).float()
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing='ij'
    )
    new_x = 2.0 * (grid_x + flow_fwd[..., 0]) / (W - 1) - 1.0
    new_y = 2.0 * (grid_y + flow_fwd[..., 1]) / (H - 1) - 1.0
    grid = torch.stack([new_x, new_y], dim=-1).unsqueeze(0)
    warped_bwd = F.grid_sample(flow_bwd_t, grid, mode='bilinear', padding_mode='border', align_corners=True)
    warped_bwd = warped_bwd.squeeze(0).permute(1, 2, 0).numpy()
    diff = np.linalg.norm(flow_fwd_np + warped_bwd, axis=-1)
    return (diff < threshold).astype(np.float32)


def fuse_frame(frames_list, center_idx, num_frames):
    center = cv2.imread(frames_list[center_idx], cv2.IMREAD_COLOR)
    H, W = center.shape[:2]
    weight_sum = np.ones((H, W, 1), dtype=np.float32) * 2.0
    pixel_sum = center.astype(np.float32) * 2.0

    for offset in [-1, 1]:
        nb_idx = center_idx + offset
        if nb_idx < 0 or nb_idx >= num_frames:
            continue
        neighbor = cv2.imread(frames_list[nb_idx], cv2.IMREAD_COLOR)

        if offset == 1:
            fwd_path = os.path.join(flow_dir, f'flow_fwd_{center_idx:05d}.npy')
            bwd_path = os.path.join(flow_dir, f'flow_bwd_{center_idx:05d}.npy')
            flow = np.load(bwd_path)  # warp next->current
        else:
            fwd_path = os.path.join(flow_dir, f'flow_fwd_{center_idx-1:05d}.npy')
            bwd_path = os.path.join(flow_dir, f'flow_bwd_{center_idx-1:05d}.npy')
            flow = np.load(fwd_path)  # warp prev->current

        if os.path.exists(fwd_path) and os.path.exists(bwd_path):
            mask = compute_occlusion_mask(np.load(fwd_path), np.load(bwd_path))
        else:
            mask = np.ones((H, W), dtype=np.float32)

        warped = warp_frame(neighbor, flow)
        weight = mask[..., np.newaxis]
        pixel_sum += warped.astype(np.float32) * weight
        weight_sum += weight

    return (pixel_sum / weight_sum).clip(0, 255).astype(np.uint8)


def main():
    frames_list = sorted(glob.glob(os.path.join(frames_dir, '*.png')))[:MAX_FRAMES]
    num_frames = len(frames_list)
    print(f"Fusing {num_frames} frames with occlusion-aware flow warping...")

    start = time.time()
    for i in range(num_frames):
        fused = fuse_frame(frames_list, i, num_frames)
        cv2.imwrite(os.path.join(fused_dir, f'frame_{i+1:05d}.png'), fused)
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start
            fps = (i + 1) / elapsed
            print(f"  [{i+1}/{num_frames}] {fps:.1f} fps, ETA: {(num_frames-i-1)/fps:.0f}s")

    print(f"\nDone! {num_frames} fused frames in {time.time()-start:.1f}s")
    print(f"Saved to {fused_dir}")


if __name__ == '__main__':
    main()
