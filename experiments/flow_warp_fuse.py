import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

"""
1080p Flow-Fusion Denoising Pipeline (Optimized)
Takes original 1080p video, uses RAFT optical flow to warp-fuse neighbors,
outputs cleaner 1080p with reduced compression artifacts.

Optimizations:
- Flow saved to disk to minimize RAM usage
- RAFT with mixed precision + autocast for VRAM efficiency
- raft-sintel model (better for live action than raft-things)
- Fusion uses float32 accumulation for precision
"""
from lib.paths import add_raft_to_path, resolve_raft_dir, DATA_DIR
add_raft_to_path()

import os, glob, time, argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from raft import RAFT
from utils.utils import InputPadder
import imageio_ffmpeg

DEVICE = 'cuda'
MAX_FRAMES = 150


def extract_frames(video_path, output_dir, max_frames=-1):
    os.makedirs(output_dir, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    import subprocess
    cmd = [ffmpeg, '-y', '-i', video_path]
    if max_frames > 0:
        cmd += ['-vframes', str(max_frames)]
    cmd += [os.path.join(output_dir, 'frame_%05d.png')]
    subprocess.run(cmd, check=True, capture_output=True)
    return sorted(glob.glob(os.path.join(output_dir, '*.png')))


def load_raft():
    args = argparse.Namespace(
        model=str(resolve_raft_dir() / "models" / "raft-sintel.pth"),
        small=False,
        mixed_precision=True,
        alternate_corr=False,
    )
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model, map_location=DEVICE, weights_only=True))
    model = model.module.to(DEVICE).eval()
    return model


FLOW_SCALE = 0.5


def load_image_tensor(path, scale=1.0):
    img = np.array(Image.open(path)).astype(np.uint8)
    if scale != 1.0:
        h, w = img.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(img).permute(2, 0, 1).float()[None].to(DEVICE)


def upscale_flow(flow_half, target_h, target_w):
    flow_up = cv2.resize(flow_half.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    flow_up *= (1.0 / FLOW_SCALE)
    return flow_up.astype(np.float16)


def compute_and_save_flows(model, frames, flow_dir):
    os.makedirs(flow_dir, exist_ok=True)
    n = len(frames) - 1
    first = np.array(Image.open(frames[0]))
    full_h, full_w = first.shape[:2]
    del first
    print(f"  Computing {n} flow pairs at {FLOW_SCALE}x res, upscaling to {full_w}x{full_h}...")
    start = time.time()
    for i in range(n):
        img1 = load_image_tensor(frames[i], scale=FLOW_SCALE)
        img2 = load_image_tensor(frames[i + 1], scale=FLOW_SCALE)
        padder = InputPadder(img1.shape)
        img1p, img2p = padder.pad(img1, img2)
        with torch.no_grad(), torch.cuda.amp.autocast():
            _, flow_fwd = model(img1p, img2p, iters=24, test_mode=True)
            _, flow_bwd = model(img2p, img1p, iters=24, test_mode=True)
        flow_fwd = padder.unpad(flow_fwd)[0].permute(1, 2, 0).cpu().numpy()
        flow_bwd = padder.unpad(flow_bwd)[0].permute(1, 2, 0).cpu().numpy()
        flow_fwd = upscale_flow(flow_fwd, full_h, full_w)
        flow_bwd = upscale_flow(flow_bwd, full_h, full_w)
        np.save(os.path.join(flow_dir, f'fwd_{i:05d}.npy'), flow_fwd)
        np.save(os.path.join(flow_dir, f'bwd_{i:05d}.npy'), flow_bwd)
        del img1, img2, img1p, img2p, flow_fwd, flow_bwd
        if i == 0:
            peak = torch.cuda.max_memory_allocated() / 1024**2
            print(f"  Peak VRAM: {peak:.0f}MB")
        if (i + 1) % 25 == 0:
            elapsed = time.time() - start
            fps = (i + 1) / elapsed
            print(f"  [{i+1}/{n}] {fps:.1f} pairs/s, ETA: {(n-i-1)/fps:.0f}s")
    elapsed = time.time() - start
    print(f"  Flow complete: {n} pairs in {elapsed:.1f}s ({n/elapsed:.1f} pairs/s)")


def warp_frame_cpu(frame_np, flow_np):
    H, W = frame_np.shape[:2]
    flow = flow_np.astype(np.float32)
    grid_y, grid_x = np.mgrid[0:H, 0:W].astype(np.float32)
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)
    return cv2.remap(frame_np, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def soft_consistency_weight(flow_fwd, flow_bwd):
    fwd = flow_fwd.astype(np.float32)
    bwd = flow_bwd.astype(np.float32)
    H, W = fwd.shape[:2]
    grid_y, grid_x = np.mgrid[0:H, 0:W].astype(np.float32)
    map_x = (grid_x + fwd[..., 0]).astype(np.float32)
    map_y = (grid_y + fwd[..., 1]).astype(np.float32)
    warped_bwd_x = cv2.remap(bwd[..., 0], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    warped_bwd_y = cv2.remap(bwd[..., 1], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    diff = np.sqrt((fwd[..., 0] + warped_bwd_x)**2 + (fwd[..., 1] + warped_bwd_y)**2)
    return np.exp(-0.5 * (diff / 0.5)**2).astype(np.float32)


CENTER_WEIGHT = 10.0
NEIGHBOR_WEIGHT = 0.5
FLOW_MAG_CUTOFF = 20.0


def fuse_frames(frames, flow_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    n = len(frames)
    print(f"  Fusing {n} frames (center={CENTER_WEIGHT}, neighbor={NEIGHBOR_WEIGHT})...")
    start = time.time()
    for i in range(n):
        center = cv2.imread(frames[i], cv2.IMREAD_COLOR)
        H, W = center.shape[:2]
        weight_sum = np.full((H, W, 1), CENTER_WEIGHT, dtype=np.float32)
        pixel_sum = center.astype(np.float32) * CENTER_WEIGHT
        for offset in [-1, 1]:
            nb_idx = i + offset
            if nb_idx < 0 or nb_idx >= n:
                continue
            neighbor = cv2.imread(frames[nb_idx], cv2.IMREAD_COLOR)
            flow_idx = min(i, nb_idx)
            fwd_path = os.path.join(flow_dir, f'fwd_{flow_idx:05d}.npy')
            bwd_path = os.path.join(flow_dir, f'bwd_{flow_idx:05d}.npy')
            if not (os.path.exists(fwd_path) and os.path.exists(bwd_path)):
                continue
            flow_fwd = np.load(fwd_path)
            flow_bwd = np.load(bwd_path)
            if offset == 1:
                flow = flow_bwd
            else:
                flow = flow_fwd
            confidence = soft_consistency_weight(flow_fwd, flow_bwd)
            flow_mag = np.sqrt(flow[..., 0].astype(np.float32)**2 + flow[..., 1].astype(np.float32)**2)
            motion_weight = np.clip(1.0 - flow_mag / FLOW_MAG_CUTOFF, 0, 1) ** 2
            warped = warp_frame_cpu(neighbor, flow)
            pixel_diff = np.mean(np.abs(warped.astype(np.float32) - center.astype(np.float32)), axis=-1)
            similarity = np.exp(-0.5 * (pixel_diff / 15.0)**2)
            combined = confidence * motion_weight * similarity * NEIGHBOR_WEIGHT
            weight = combined[..., np.newaxis]
            pixel_sum += warped.astype(np.float32) * weight
            weight_sum += weight
        fused = (pixel_sum / weight_sum).clip(0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, f'frame_{i+1:05d}.png'), fused)
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start
            print(f"  [{i+1}/{n}] {(i+1)/elapsed:.1f} fps")
    print(f"  Fusion complete in {time.time()-start:.1f}s")


def frames_to_video(frames_dir, output_path, fps=23.976):
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    import subprocess
    subprocess.run([
        ffmpeg, '-y', '-framerate', str(fps),
        '-i', os.path.join(frames_dir, 'frame_%05d.png'),
        '-c:v', 'libx264', '-crf', '14', '-pix_fmt', 'yuv420p',
        output_path
    ], check=True, capture_output=True)


def main():
    data_dir = str(DATA_DIR)
    src_video = os.path.join(data_dir, 'clip_1080p.mp4')
    frames_dir = os.path.join(data_dir, 'frames_1080p_src')
    flow_dir = os.path.join(data_dir, 'flow_1080p')
    output_dir = os.path.join(data_dir, 'frames_1080p_denoised')
    print("=" * 60)
    print("Step 1: Extract 1080p frames")
    print("=" * 60)
    if os.path.exists(frames_dir) and len(glob.glob(os.path.join(frames_dir, '*.png'))) > 0:
        frames = sorted(glob.glob(os.path.join(frames_dir, '*.png')))
        if MAX_FRAMES > 0:
            frames = frames[:MAX_FRAMES]
        print(f"  Using {len(frames)} existing frames")
    else:
        frames = extract_frames(src_video, frames_dir, max_frames=MAX_FRAMES)
    print("\n" + "=" * 60)
    print("Step 2: RAFT optical flow at 1080p")
    print("=" * 60)
    expected_flows = (len(frames) - 1) * 2
    existing_flows = len(glob.glob(os.path.join(flow_dir, '*.npy'))) if os.path.exists(flow_dir) else 0
    if existing_flows >= expected_flows:
        print(f"  Reusing {existing_flows} existing flow files")
    else:
        raft_model = load_raft()
        torch.cuda.reset_peak_memory_stats()
        compute_and_save_flows(raft_model, frames, flow_dir)
        del raft_model
        torch.cuda.empty_cache()
    print("\n" + "=" * 60)
    print("Step 3: Occlusion-aware fusion")
    print("=" * 60)
    fuse_frames(frames, flow_dir, output_dir)
    print("\n" + "=" * 60)
    print("Step 4: Encode output")
    print("=" * 60)
    output_video = os.path.join(data_dir, 'clip_1080p_denoised.mp4')
    frames_to_video(output_dir, output_video)
    print(f"  Output: {output_video}")
    print(f"  Original: {src_video}")
    print("\nDone! Compare the two videos side by side.")


if __name__ == '__main__':
    main()
