"""Test A: Warp neighboring frames using RAFT flow, fuse, then Real-ESRGAN upscale.
Tests whether temporal information via optical flow improves single-image SR."""
import os, glob, time
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

data_dir = r'C:\Users\sean\src\upscale-experiment\data'
frames_dir = os.path.join(data_dir, 'frames_480p')
flow_dir = os.path.join(data_dir, 'flow_npy')
output_dir = os.path.join(data_dir, 'frames_warp_fuse_sr')
fused_dir = os.path.join(data_dir, 'frames_fused_480p')  # save fused frames for inspection
os.makedirs(output_dir, exist_ok=True)
os.makedirs(fused_dir, exist_ok=True)

WINDOW = 2  # use +/- 2 neighbors
MAX_FRAMES = 150


def warp_frame(frame_tensor, flow_np):
    """Warp a frame using optical flow via grid_sample.
    frame_tensor: (H, W, 3) numpy uint8
    flow_np: (H, W, 2) numpy float32
    Returns: (H, W, 3) numpy uint8
    """
    H, W = frame_tensor.shape[:2]
    frame = torch.from_numpy(frame_tensor).float().permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    flow = torch.from_numpy(flow_np).float()  # (H,W,2)

    # Build sampling grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing='ij'
    )
    # flow[...,0] is horizontal (x), flow[...,1] is vertical (y)
    new_x = grid_x + flow[..., 0]
    new_y = grid_y + flow[..., 1]

    # Normalize to [-1, 1]
    new_x = 2.0 * new_x / (W - 1) - 1.0
    new_y = 2.0 * new_y / (H - 1) - 1.0

    grid = torch.stack([new_x, new_y], dim=-1).unsqueeze(0)  # (1,H,W,2)
    warped = F.grid_sample(frame, grid, mode='bilinear', padding_mode='border', align_corners=True)
    return warped.squeeze(0).permute(1, 2, 0).clamp(0, 255).byte().numpy()


def compute_occlusion_mask(flow_fwd, flow_bwd, threshold=1.5):
    """Forward-backward consistency check. Returns mask where 1=consistent, 0=occluded."""
    H, W = flow_fwd.shape[:2]
    # Warp backward flow using forward flow
    flow_bwd_t = torch.from_numpy(flow_bwd).float().permute(2, 0, 1).unsqueeze(0)
    flow_fwd_t = torch.from_numpy(flow_fwd).float()

    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing='ij'
    )
    new_x = grid_x + flow_fwd_t[..., 0]
    new_y = grid_y + flow_fwd_t[..., 1]
    new_x = 2.0 * new_x / (W - 1) - 1.0
    new_y = 2.0 * new_y / (H - 1) - 1.0
    grid = torch.stack([new_x, new_y], dim=-1).unsqueeze(0)

    warped_bwd = F.grid_sample(flow_bwd_t, grid, mode='bilinear', padding_mode='border', align_corners=True)
    warped_bwd = warped_bwd.squeeze(0).permute(1, 2, 0).numpy()

    # Check consistency: fwd + warped_bwd should be ~0
    diff = np.linalg.norm(flow_fwd + warped_bwd, axis=-1)
    mask = (diff < threshold).astype(np.float32)
    return mask


def fuse_frames(frames_list, center_idx, flow_dir, num_frames):
    """Warp neighbors to center frame and fuse with occlusion-aware weighting."""
    center = cv2.imread(frames_list[center_idx], cv2.IMREAD_COLOR)
    H, W = center.shape[:2]

    # Accumulate weighted sum
    weight_sum = np.ones((H, W, 1), dtype=np.float32) * 2.0  # center gets weight 2
    pixel_sum = center.astype(np.float32) * 2.0

    for offset in range(-WINDOW, WINDOW + 1):
        if offset == 0:
            continue
        neighbor_idx = center_idx + offset
        if neighbor_idx < 0 or neighbor_idx >= num_frames:
            continue

        neighbor = cv2.imread(frames_list[neighbor_idx], cv2.IMREAD_COLOR)

        # Chain flows from neighbor to center
        # For simplicity, use direct flow if |offset|==1, otherwise chain
        if offset == 1:
            # Need flow from frame center_idx+1 -> center_idx (backward flow at center_idx)
            flow_path = os.path.join(flow_dir, f'flow_bwd_{center_idx:05d}.npy')
            if not os.path.exists(flow_path):
                continue
            flow = np.load(flow_path)
        elif offset == -1:
            # Need flow from frame center_idx-1 -> center_idx (forward flow at center_idx-1)
            flow_path = os.path.join(flow_dir, f'flow_fwd_{center_idx-1:05d}.npy')
            if not os.path.exists(flow_path):
                continue
            flow = np.load(flow_path)
        else:
            # For larger offsets, skip (chaining flow is error-prone)
            # Just use +/-1 for now
            continue

        # Compute occlusion mask
        if offset == 1:
            fwd_path = os.path.join(flow_dir, f'flow_fwd_{center_idx:05d}.npy')
            bwd_path = os.path.join(flow_dir, f'flow_bwd_{center_idx:05d}.npy')
        else:  # offset == -1
            fwd_path = os.path.join(flow_dir, f'flow_fwd_{center_idx-1:05d}.npy')
            bwd_path = os.path.join(flow_dir, f'flow_bwd_{center_idx-1:05d}.npy')

        if os.path.exists(fwd_path) and os.path.exists(bwd_path):
            mask = compute_occlusion_mask(np.load(fwd_path), np.load(bwd_path))
        else:
            mask = np.ones((H, W), dtype=np.float32)

        warped = warp_frame(neighbor, flow)
        weight = mask[..., np.newaxis]  # (H,W,1)
        pixel_sum += warped.astype(np.float32) * weight
        weight_sum += weight

    fused = (pixel_sum / weight_sum).clip(0, 255).astype(np.uint8)
    return fused


def main():
    frames_list = sorted(glob.glob(os.path.join(frames_dir, '*.png')))[:MAX_FRAMES]
    num_frames = len(frames_list)
    print(f"Fusing {num_frames} frames with RAFT flow (window={WINDOW})...")

    # Phase 1: Generate fused frames (CPU, no GPU needed)
    fused_paths = []
    start = time.time()
    for i in range(num_frames):
        fused = fuse_frames(frames_list, i, flow_dir, num_frames)
        fused_path = os.path.join(fused_dir, f'frame_{i+1:05d}.png')
        cv2.imwrite(fused_path, fused)
        fused_paths.append(fused_path)
        if (i + 1) % 50 == 0:
            print(f"  Fused [{i+1}/{num_frames}]")
    print(f"Fusion done in {time.time()-start:.1f}s")

    # Phase 2: Upscale fused frames with Real-ESRGAN (GPU)
    print("\nLoading Real-ESRGAN...")
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4,
        model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
        model=model,
        tile=384,
        tile_pad=10,
        pre_pad=0,
        half=True,
        gpu_id=0,
    )

    print(f"Upscaling {num_frames} fused frames...")
    start = time.time()
    for i, fused_path in enumerate(fused_paths):
        img = cv2.imread(fused_path, cv2.IMREAD_COLOR)
        output, _ = upsampler.enhance(img, outscale=4)
        output = cv2.resize(output, (1920, 1080), interpolation=cv2.INTER_LANCZOS4)
        out_path = os.path.join(output_dir, f'frame_{i+1:05d}.png')
        cv2.imwrite(out_path, output)
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start
            fps = (i + 1) / elapsed
            eta = (num_frames - i - 1) / fps
            print(f"  SR [{i+1}/{num_frames}] {fps:.2f} fps, ETA: {eta:.0f}s")

    elapsed = time.time() - start
    print(f"\nDone! {num_frames} frames upscaled in {elapsed:.1f}s")


if __name__ == '__main__':
    main()
