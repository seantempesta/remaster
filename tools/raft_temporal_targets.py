"""Generate temporally-averaged targets using RAFT optical flow alignment.

For each frame in a video clip, align N neighboring frames using RAFT optical
flow, then take the median of aligned observations to produce a clean, sharp
target. This recovers real detail from multiple noisy/compressed observations
without hallucinating content.

Algorithm:
    For each frame t:
        Load frames [t-W, ..., t, ..., t+W]
        For each neighbor n != t:
            flow_fwd = RAFT(frame_t, frame_n)    # where does t content appear in n?
            flow_bwd = RAFT(frame_n, frame_t)    # where does n content appear in t?
            warped = warp(frame_n, flow_fwd)      # pull n's pixels to t's grid
            confidence = forward_backward_consistency(flow_fwd, flow_bwd)
            Store warped + confidence on CPU
        For each pixel:
            target[x,y] = weighted_median(valid_observations)
        Save target as PNG

Usage:
    python tools/raft_temporal_targets.py \
        -i data/archive/firefly_s01e08_30s.mkv \
        -o data/archive/raft_targets/ \
        --window 4 \
        --max-frames 100 \
        --comparison
"""

import argparse
import gc
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "reference-code", "RAFT", "core"))

from lib.ffmpeg_utils import get_ffmpeg, get_ffprobe, get_video_info

# RAFT imports (from reference-code/RAFT/core)
from raft import RAFT
from utils.utils import InputPadder


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RAFT_THINGS_PATH = os.path.join(
    PROJECT_ROOT, "reference-code", "RAFT", "models", "raft-things.pth"
)
RAFT_SMALL_PATH = os.path.join(
    PROJECT_ROOT, "reference-code", "RAFT", "models", "raft-small.pth"
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Forward-backward consistency threshold (in pixels).
# Pairs of flow vectors that disagree by more than this are marked as occluded.
FB_CONSISTENCY_THRESH = 1.5

# Minimum number of valid (non-occluded) observations to use the median.
# If fewer are available, fall back to the center frame.
MIN_VALID_OBS = 2

# RAFT iterations -- more = better flow, slower. 20 is the demo default.
RAFT_ITERS = 20

# Use RAFT-Small by default (fits easily in 6GB VRAM)
USE_SMALL = True


# ---------------------------------------------------------------------------
# GPU memory helpers
# ---------------------------------------------------------------------------
def gpu_mem_str():
    """Return a short string showing current GPU memory usage."""
    if not torch.cuda.is_available():
        return "CPU"
    alloc = torch.cuda.memory_allocated() / 1e6
    reserved = torch.cuda.memory_reserved() / 1e6
    return f"GPU {alloc:.0f}/{reserved:.0f} MB (alloc/reserved)"


# ---------------------------------------------------------------------------
# RAFT model loading
# ---------------------------------------------------------------------------
def load_raft_model(small=None, mixed_precision=True):
    """Load RAFT optical flow model.

    Args:
        small: Use RAFT-Small (True) or RAFT-Things (False). Defaults to USE_SMALL.
        mixed_precision: Enable fp16 mixed precision for inference.
    """
    if small is None:
        small = USE_SMALL

    model_path = RAFT_SMALL_PATH if small else RAFT_THINGS_PATH
    model_name = "RAFT-Small" if small else "RAFT-Things"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"{model_name} weights not found at {model_path}. "
            "Download from https://github.com/princeton-vl/RAFT"
        )

    # RAFT expects an args namespace with 'in' operator support
    class RAFTArgs:
        def __init__(self):
            self.small = small
            self.mixed_precision = mixed_precision
            self.alternate_corr = False
            self.dropout = 0.0

        def __contains__(self, key):
            return hasattr(self, key)

    args = RAFTArgs()
    model = torch.nn.DataParallel(RAFT(args))
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model = model.module
    model.to(DEVICE)
    model.eval()
    print(f"[RAFT] Loaded {model_name} from {model_path} on {DEVICE}", flush=True)
    print(f"[RAFT] {gpu_mem_str()}", flush=True)
    return model


# ---------------------------------------------------------------------------
# Frame extraction via ffmpeg
# ---------------------------------------------------------------------------
def extract_frames_to_numpy(video_path, max_frames=None):
    """Extract all frames from video as a list of uint8 numpy arrays [H,W,3].

    Uses ffmpeg to pipe raw RGB frames. Stores all frames in CPU RAM.
    For a 30s 1080p clip (~720 frames), this is about 720 * 6MB = 4.3GB RAM.
    """
    w, h, fps, total_frames, duration = get_video_info(video_path)
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)

    print(
        f"[extract] Video: {w}x{h} @ {fps:.3f} fps, extracting {total_frames} frames",
        flush=True,
    )

    ffmpeg = get_ffmpeg()
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-i", video_path,
        "-pix_fmt", "rgb24",
        "-f", "rawvideo",
    ]
    if max_frames is not None:
        cmd += ["-frames:v", str(max_frames)]
    cmd += ["pipe:1"]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    frame_size = w * h * 3
    frames = []
    while True:
        raw = proc.stdout.read(frame_size)
        if len(raw) < frame_size:
            break
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
        frames.append(frame)
        if max_frames is not None and len(frames) >= max_frames:
            break

    proc.stdout.close()
    proc.wait()
    print(
        f"[extract] Got {len(frames)} frames ({len(frames) * frame_size / 1e9:.2f} GB RAM)",
        flush=True,
    )
    return frames, w, h, fps


# ---------------------------------------------------------------------------
# Optical flow utilities
# ---------------------------------------------------------------------------
def numpy_to_raft_tensor(frame_np):
    """Convert HWC uint8 numpy to 1CHW float32 GPU tensor (0-255 range, as RAFT expects)."""
    return torch.from_numpy(frame_np).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)


FLOW_HALF_RES = False  # Set via --half-res CLI flag


@torch.no_grad()
def compute_flow_pair(model, frame1_np, frame2_np, prefilter=False,
                      prefilter_sigma=2.0):
    """Compute optical flow from frame1 to frame2 using RAFT.

    Full-res by default (RAFT-Small fits in 6GB). Falls back to half-res
    with --half-res flag if VRAM is tight.

    Args:
        model: RAFT model
        frame1_np: [H, W, 3] uint8 numpy array (source frame)
        frame2_np: [H, W, 3] uint8 numpy array (target frame)
        prefilter: if True, low-pass filter inputs to RAFT for cleaner flow
        prefilter_sigma: Gaussian sigma for pre-filter (default 2.0)

    Returns flow tensor [1, 2, H, W] on CPU at full resolution.
    """
    H, W = frame1_np.shape[:2]

    # Optionally pre-filter for cleaner feature matching
    if prefilter:
        raft_in1 = prefilter_for_flow(frame1_np, sigma=prefilter_sigma)
        raft_in2 = prefilter_for_flow(frame2_np, sigma=prefilter_sigma)
    else:
        raft_in1, raft_in2 = frame1_np, frame2_np

    # Scale factor: 1.0 = full res, 0.5 = half res, 0.75 = 3/4 res
    # Auto-select based on image size and available VRAM
    if FLOW_HALF_RES:
        scale = 0.5
    else:
        # 3/4 res for 1080p (fits in 6GB), full res for smaller
        scale = 0.75 if (H * W) > 1000000 else 1.0

    if scale < 1.0:
        hs, ws = int(H * scale), int(W * scale)
        f1 = cv2.resize(raft_in1, (ws, hs), interpolation=cv2.INTER_AREA)
        f2 = cv2.resize(raft_in2, (ws, hs), interpolation=cv2.INTER_AREA)
    else:
        f1, f2 = raft_in1, raft_in2
        scale = 1.0

    img1 = numpy_to_raft_tensor(f1)
    img2 = numpy_to_raft_tensor(f2)

    padder = InputPadder(img1.shape)
    img1_p, img2_p = padder.pad(img1, img2)

    _, flow_up = model(img1_p, img2_p, iters=RAFT_ITERS, test_mode=True)
    flow_up = padder.unpad(flow_up)

    flow_cpu = flow_up.cpu()
    del img1, img2, img1_p, img2_p, flow_up
    torch.cuda.empty_cache()

    if scale < 1.0:
        flow_cpu = F.interpolate(
            flow_cpu, size=(H, W), mode="bilinear", align_corners=False
        ) / scale  # scale flow magnitudes inversely

    return flow_cpu


@torch.no_grad()
def compute_flow_batch(model, pairs, max_batch=8):
    """Compute optical flow for multiple frame pairs in a batch.

    Processes pairs in batches for better GPU utilization. Each pair is
    (frame1_np, frame2_np) where frames are [H, W, 3] uint8 numpy arrays.
    All frames must have the same resolution.

    Args:
        model: RAFT model
        pairs: list of (frame1_np, frame2_np) tuples
        max_batch: max pairs per batch (limited by VRAM)

    Returns:
        list of [1, 2, H, W] flow tensors on GPU
    """
    if not pairs:
        return []

    results = []
    for batch_start in range(0, len(pairs), max_batch):
        batch_pairs = pairs[batch_start:batch_start + max_batch]
        B = len(batch_pairs)

        # Stack all frame1s and frame2s into batches
        imgs1 = torch.cat([numpy_to_raft_tensor(p[0]) for p in batch_pairs], dim=0)  # [B, 3, H, W]
        imgs2 = torch.cat([numpy_to_raft_tensor(p[1]) for p in batch_pairs], dim=0)  # [B, 3, H, W]

        padder = InputPadder(imgs1.shape)
        imgs1_p, imgs2_p = padder.pad(imgs1, imgs2)

        _, flow_up = model(imgs1_p, imgs2_p, iters=RAFT_ITERS, test_mode=True)
        flow_up = padder.unpad(flow_up)  # [B, 2, H, W]

        # Split batch back into individual flows (keep on GPU)
        for i in range(B):
            results.append(flow_up[i:i+1])  # [1, 2, H, W] on GPU

        del imgs1, imgs2, imgs1_p, imgs2_p, flow_up

    return results


def warp_frame(frame_np, flow_cpu):
    """Warp a frame using optical flow via grid_sample.

    Args:
        frame_np: [H, W, 3] uint8 numpy array (the neighbor frame to warp)
        flow_cpu: [1, 2, H, W] float tensor on CPU (flow from center to neighbor)

    Returns:
        warped: [H, W, 3] float32 numpy array (0-255 range)
        in_bounds: [H, W] bool numpy array (True where warped pixels are valid)
    """
    H, W = frame_np.shape[:2]

    # Build sampling grid: pixel coords + flow
    # flow[0] = dx, flow[1] = dy
    gy, gx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing="ij",
    )
    # flow_cpu is [1, 2, H, W] -- squeeze batch dim
    flow_sq = flow_cpu.squeeze(0)  # [2, H, W]
    sample_x = gx + flow_sq[0]  # [H, W]
    sample_y = gy + flow_sq[1]  # [H, W]

    # Normalize to [-1, 1] for grid_sample
    norm_x = 2.0 * sample_x / (W - 1) - 1.0
    norm_y = 2.0 * sample_y / (H - 1) - 1.0
    grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0)  # [1, H, W, 2]

    # Convert frame to tensor for grid_sample
    frame_t = torch.from_numpy(frame_np).permute(2, 0, 1).float().unsqueeze(0)  # [1, 3, H, W]

    warped = F.grid_sample(frame_t, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    warped = warped.squeeze(0).permute(1, 2, 0).numpy()  # [H, W, 3]

    # In-bounds mask: check if sample coords fall within the frame
    in_bounds = (
        (sample_x >= 0) & (sample_x <= W - 1) &
        (sample_y >= 0) & (sample_y <= H - 1)
    ).numpy()

    return warped, in_bounds


def warp_frame_gpu(frame_np, flow_gpu):
    """Warp a frame on GPU, returning GPU tensors.

    Like warp_frame() but keeps everything on GPU to avoid CPU round-trips.

    Args:
        frame_np: [H, W, 3] uint8 numpy array (the neighbor frame to warp)
        flow_gpu: [1, 2, H, W] float tensor on GPU (flow from center to neighbor)

    Returns:
        warped: [H, W, 3] float32 GPU tensor (0-255 range)
        in_bounds: [H, W] bool GPU tensor (True where warped pixels are valid)
    """
    H, W = frame_np.shape[:2]
    device = flow_gpu.device

    # Build sampling grid on GPU
    gy, gx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing="ij",
    )
    flow_sq = flow_gpu.squeeze(0)  # [2, H, W]
    sample_x = gx + flow_sq[0]
    sample_y = gy + flow_sq[1]

    norm_x = 2.0 * sample_x / (W - 1) - 1.0
    norm_y = 2.0 * sample_y / (H - 1) - 1.0
    grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0)

    frame_t = torch.from_numpy(frame_np).permute(2, 0, 1).float().unsqueeze(0).to(device)
    warped = F.grid_sample(frame_t, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    warped = warped.squeeze(0).permute(1, 2, 0)  # [H, W, 3] on GPU

    in_bounds = (
        (sample_x >= 0) & (sample_x <= W - 1) &
        (sample_y >= 0) & (sample_y <= H - 1)
    )

    return warped, in_bounds


def compute_occlusion_mask(flow_fwd_cpu, flow_bwd_cpu):
    """Forward-backward consistency check for occlusion detection.

    For each pixel p in frame1:
        1. Follow flow_fwd to find corresponding location p' in frame2
        2. At p', sample flow_bwd to get the return vector
        3. If ||flow_fwd(p) + flow_bwd(p')|| > threshold, pixel is occluded

    Args:
        flow_fwd_cpu: [1, 2, H, W] flow from frame1 -> frame2 (CPU)
        flow_bwd_cpu: [1, 2, H, W] flow from frame2 -> frame1 (CPU)

    Returns:
        valid_mask: [H, W] bool numpy array (True = not occluded, safe to use)
    """
    H, W = flow_fwd_cpu.shape[2], flow_fwd_cpu.shape[3]

    # Sample backward flow at the locations pointed to by forward flow
    gy, gx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing="ij",
    )
    flow_fwd = flow_fwd_cpu.squeeze(0)  # [2, H, W]
    target_x = gx + flow_fwd[0]
    target_y = gy + flow_fwd[1]

    # Normalize for grid_sample
    norm_x = 2.0 * target_x / (W - 1) - 1.0
    norm_y = 2.0 * target_y / (H - 1) - 1.0
    grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0)  # [1, H, W, 2]

    # Sample backward flow at forward-warped positions
    flow_bwd_sampled = F.grid_sample(
        flow_bwd_cpu, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )  # [1, 2, H, W]

    # Round-trip error: fwd(p) + bwd(p') should be ~0 for non-occluded pixels
    round_trip = flow_fwd_cpu + flow_bwd_sampled  # [1, 2, H, W]
    error = torch.norm(round_trip, dim=1).squeeze(0)  # [H, W]

    valid_mask = (error < FB_CONSISTENCY_THRESH).numpy()
    return valid_mask


def compute_occlusion_mask_gpu(flow_fwd_gpu, flow_bwd_gpu):
    """Forward-backward consistency check on GPU, returning GPU tensor.

    Same algorithm as compute_occlusion_mask() but keeps everything on GPU.

    Args:
        flow_fwd_gpu: [1, 2, H, W] flow from frame1 -> frame2 (GPU)
        flow_bwd_gpu: [1, 2, H, W] flow from frame2 -> frame1 (GPU)

    Returns:
        valid_mask: [H, W] bool GPU tensor (True = not occluded)
    """
    device = flow_fwd_gpu.device
    H, W = flow_fwd_gpu.shape[2], flow_fwd_gpu.shape[3]

    gy, gx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing="ij",
    )
    flow_fwd = flow_fwd_gpu.squeeze(0)
    target_x = gx + flow_fwd[0]
    target_y = gy + flow_fwd[1]

    norm_x = 2.0 * target_x / (W - 1) - 1.0
    norm_y = 2.0 * target_y / (H - 1) - 1.0
    grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0)

    flow_bwd_sampled = F.grid_sample(
        flow_bwd_gpu, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )

    round_trip = flow_fwd_gpu + flow_bwd_sampled
    error = torch.norm(round_trip, dim=1).squeeze(0)
    valid_mask = error < FB_CONSISTENCY_THRESH
    return valid_mask


# ---------------------------------------------------------------------------
# Temporal median computation
# ---------------------------------------------------------------------------
def compute_temporal_median(center_np, warped_list, mask_list):
    """Compute weighted median of aligned observations.

    Args:
        center_np: [H, W, 3] uint8, the center frame
        warped_list: list of [H, W, 3] float32 arrays (warped neighbors)
        mask_list: list of [H, W] bool arrays (valid pixels)

    Returns:
        result: [H, W, 3] uint8 numpy array
    """
    H, W, C = center_np.shape
    center_f = center_np.astype(np.float32)

    # Stack all observations: center + warped neighbors
    # Center frame is always valid everywhere
    all_obs = [center_f] + warped_list
    all_masks = [np.ones((H, W), dtype=bool)] + mask_list

    # Stack into arrays for vectorized processing
    obs_stack = np.stack(all_obs, axis=0)   # [N, H, W, 3]
    mask_stack = np.stack(all_masks, axis=0)  # [N, H, W]

    N = obs_stack.shape[0]
    result = np.empty((H, W, C), dtype=np.float32)

    # Count valid observations per pixel
    valid_count = mask_stack.sum(axis=0)  # [H, W]

    # For pixels with enough valid observations, compute per-pixel median.
    # For efficiency, process channel by channel.
    for c in range(C):
        channel_data = obs_stack[:, :, :, c]  # [N, H, W]

        # Set invalid observations to NaN so nanmedian ignores them
        channel_masked = np.where(mask_stack, channel_data, np.nan)

        # Compute nanmedian along the observation axis
        with np.errstate(all="ignore"):
            median_vals = np.nanmedian(channel_masked, axis=0)  # [H, W]

        # Fall back to center frame where we lack enough valid observations
        fallback = valid_count < MIN_VALID_OBS
        median_vals[fallback] = center_f[fallback, c]

        # Handle any remaining NaNs (shouldn't happen, but be safe)
        nan_mask = np.isnan(median_vals)
        median_vals[nan_mask] = center_f[nan_mask, c]

        result[:, :, c] = median_vals

    return np.clip(result, 0, 255).astype(np.uint8)


def compute_temporal_median_gpu(center_np, warped_gpu_list, mask_gpu_list):
    """Compute weighted median of aligned observations on GPU.

    GPU-accelerated version of compute_temporal_median(). Uses torch.nanmedian
    on GPU tensors instead of np.nanmedian on CPU -- typically 10-100x faster.

    Args:
        center_np: [H, W, 3] uint8, the center frame
        warped_gpu_list: list of [H, W, 3] float32 GPU tensors (warped neighbors)
        mask_gpu_list: list of [H, W] bool GPU tensors (valid pixels)

    Returns:
        result: [H, W, 3] uint8 numpy array
    """
    device = warped_gpu_list[0].device if warped_gpu_list else DEVICE
    H, W, C = center_np.shape

    center_t = torch.from_numpy(center_np).float().to(device)  # [H, W, 3]

    # Stack: center (always valid) + warped neighbors
    all_obs = [center_t] + warped_gpu_list  # list of [H, W, 3] GPU tensors
    all_masks = [torch.ones(H, W, dtype=torch.bool, device=device)] + mask_gpu_list

    obs_stack = torch.stack(all_obs, dim=0)   # [N, H, W, 3]
    mask_stack = torch.stack(all_masks, dim=0)  # [N, H, W]

    valid_count = mask_stack.sum(dim=0)  # [H, W]

    # Set invalid observations to NaN for nanmedian
    # Expand mask to match channel dim: [N, H, W] -> [N, H, W, 1]
    mask_expanded = mask_stack.unsqueeze(-1).expand_as(obs_stack)
    obs_stack = torch.where(mask_expanded, obs_stack, torch.tensor(float('nan'), device=device))

    # torch.nanmedian along observation axis (dim=0)
    # Returns (values, indices) -- we only need values
    result, _ = torch.nanmedian(obs_stack, dim=0)  # [H, W, 3]

    # Fall back to center frame where we lack enough valid observations
    fallback = (valid_count < MIN_VALID_OBS).unsqueeze(-1).expand_as(result)
    result = torch.where(fallback, center_t, result)

    # Handle any remaining NaNs
    nan_mask = torch.isnan(result)
    result = torch.where(nan_mask, center_t, result)

    return result.clamp(0, 255).to(torch.uint8).cpu().numpy()


def compute_temporal_fft(center_np, warped_list, mask_list):
    """Denoise via FFT magnitude averaging across aligned frames (simple version).

    For each channel: FFT all observations, average magnitudes (noise cancels
    in frequency domain), keep reference phase from center frame, IFFT back.

    This is more principled than spatial median for Gaussian-like noise.
    Superseded by compute_temporal_fft_confidence() when --fft-denoise is used.
    """
    H, W, C = center_np.shape
    center_f = center_np.astype(np.float32)

    all_obs = [center_f] + warped_list
    all_masks = [np.ones((H, W), dtype=bool)] + mask_list

    obs_stack = np.stack(all_obs, axis=0)   # [N, H, W, 3]
    mask_stack = np.stack(all_masks, axis=0)  # [N, H, W]
    N = obs_stack.shape[0]

    result = np.empty((H, W, C), dtype=np.float32)
    valid_count = mask_stack.sum(axis=0)  # [H, W]

    for c in range(C):
        # FFT the center frame (reference phase)
        center_fft = np.fft.rfft2(center_f[:, :, c])
        ref_phase = np.angle(center_fft)

        # Average magnitudes across valid observations
        mag_sum = np.abs(center_fft).copy()
        mag_count = np.ones_like(mag_sum)

        for i in range(1, N):
            obs = obs_stack[i, :, :, c].copy()
            mask = mask_stack[i]
            # For invalid pixels, substitute center frame to avoid artifacts
            obs[~mask] = center_f[~mask, c]
            obs_fft = np.fft.rfft2(obs)
            mag_sum += np.abs(obs_fft)
            mag_count += 1.0

        avg_mag = mag_sum / mag_count

        # Reconstruct: averaged magnitude + reference phase
        clean_fft = avg_mag * np.exp(1j * ref_phase)
        clean = np.fft.irfft2(clean_fft, s=(H, W))

        # Fall back to center where not enough valid observations
        fallback = valid_count < MIN_VALID_OBS
        clean[fallback] = center_f[fallback, c]

        result[:, :, c] = clean

    return np.clip(result, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Component 1: Pre-filter for cleaner RAFT flow
# ---------------------------------------------------------------------------
def prefilter_for_flow(frame_np, sigma=2.0):
    """Low-pass filter for cleaner RAFT flow computation.

    Removes high-frequency noise so RAFT matches real features, not noise.
    The filtered output is only used as RAFT input -- original frames are
    still warped and averaged.

    Args:
        frame_np: [H, W, 3] uint8 numpy array
        sigma: Gaussian blur sigma (default 2.0)

    Returns:
        blurred: [H, W, 3] uint8 numpy array
    """
    blurred = cv2.GaussianBlur(
        frame_np.astype(np.float32), (0, 0), sigmaX=sigma
    )
    return np.clip(blurred, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Component 2: FFT confidence-weighted averaging
# ---------------------------------------------------------------------------
def compute_temporal_fft_confidence(center_np, warped_list, mask_list):
    """FFT averaging with per-frequency confidence weighting.

    For each frequency bin:
        - Compute magnitude across all aligned frames
        - Low variance = consistent = real signal -> high weight
        - High variance = noise -> low weight
        - Weighted average of complex FFT values
        - Keep reference phase from center frame for stability

    This improves on compute_temporal_fft() by down-weighting frequency
    bins where aligned frames disagree (likely noise or misalignment).

    Args:
        center_np: [H, W, 3] uint8, the center frame
        warped_list: list of [H, W, 3] float32 arrays (warped neighbors)
        mask_list: list of [H, W] bool arrays (valid pixels)

    Returns:
        result: [H, W, 3] uint8 numpy array
    """
    H, W, C = center_np.shape
    center_f = center_np.astype(np.float32)

    all_obs = [center_f] + warped_list
    all_masks = [np.ones((H, W), dtype=bool)] + mask_list

    obs_stack = np.stack(all_obs, axis=0)   # [N, H, W, 3]
    mask_stack = np.stack(all_masks, axis=0)  # [N, H, W]
    N = obs_stack.shape[0]

    result = np.empty((H, W, C), dtype=np.float32)
    valid_count = mask_stack.sum(axis=0)  # [H, W]

    for c in range(C):
        # FFT the center frame (reference phase)
        center_fft = np.fft.rfft2(center_f[:, :, c])
        ref_phase = np.angle(center_fft)
        fft_shape = center_fft.shape  # (H, W//2+1)

        # Collect FFT magnitudes from all observations
        all_mags = []
        all_ffts = []
        for i in range(N):
            obs = obs_stack[i, :, :, c].copy()
            mask = mask_stack[i]
            # For invalid pixels, substitute center frame to avoid artifacts
            obs[~mask] = center_f[~mask, c]
            obs_fft = np.fft.rfft2(obs)
            all_ffts.append(obs_fft)
            all_mags.append(np.abs(obs_fft))

        mag_stack = np.stack(all_mags, axis=0)  # [N, H, W//2+1]

        # Per-frequency confidence: inverse of normalized variance
        # High variance across frames -> noise -> low confidence
        mag_mean = mag_stack.mean(axis=0)  # [H, W//2+1]
        mag_var = mag_stack.var(axis=0)    # [H, W//2+1]

        # Confidence = 1 / (1 + var/mean^2) -- coefficient of variation squared
        # Avoid division by zero for DC or flat regions
        confidence = 1.0 / (1.0 + mag_var / (mag_mean ** 2 + 1e-8))
        # confidence is in [0, 1]: 1.0 = perfectly consistent, 0.0 = all noise

        # Weighted average of magnitudes (weight = confidence, but here all
        # observations are averaged equally -- confidence modulates how much
        # the averaged result replaces the center frame)
        avg_mag = mag_stack.mean(axis=0)

        # Blend: confident frequencies use averaged magnitude, uncertain ones
        # keep the center frame's magnitude
        center_mag = np.abs(center_fft)
        blended_mag = confidence * avg_mag + (1.0 - confidence) * center_mag

        # Reconstruct: blended magnitude + reference phase
        clean_fft = blended_mag * np.exp(1j * ref_phase)
        clean = np.fft.irfft2(clean_fft, s=(H, W))

        # Fall back to center where not enough valid observations
        fallback = valid_count < MIN_VALID_OBS
        clean[fallback] = center_f[fallback, c]

        result[:, :, c] = clean

    return np.clip(result, 0, 255).astype(np.uint8)


def compute_wiener_sharpen(center_np, warped_list, mask_list,
                            wiener_strength=1.0, sharpen_strength=1.0,
                            sharpen_cutoff=0.3):
    """Wiener filter the center frame using temporal statistics, then sharpen.

    Unlike averaging approaches, this FILTERS THE CENTER FRAME based on
    per-frequency SNR estimated from temporal consistency. Then boosts
    high frequencies for sharpening.

    Step 1 (Denoise): For each frequency bin, estimate SNR from variance
    across aligned frames. Apply Wiener gain: keep signal, suppress noise.
    Step 2 (Sharpen): Boost high frequencies in the cleaned result.

    Args:
        center_np: [H, W, 3] uint8, the center frame
        warped_list: list of [H, W, 3] float32 arrays (warped neighbors)
        mask_list: list of [H, W] bool arrays (valid pixels)
        wiener_strength: 0-2, how aggressively to denoise (1.0 = standard)
        sharpen_strength: 0-3, how much to boost high frequencies (0 = none)
        sharpen_cutoff: 0-1, fraction of max frequency where boost starts

    Returns:
        result: [H, W, 3] uint8 numpy array
    """
    H, W, C = center_np.shape
    center_f = center_np.astype(np.float32)

    all_obs = [center_f] + warped_list
    all_masks = [np.ones((H, W), dtype=bool)] + mask_list

    obs_stack = np.stack(all_obs, axis=0)   # [N, H, W, 3]
    mask_stack = np.stack(all_masks, axis=0)  # [N, H, W]
    N = obs_stack.shape[0]

    result = np.empty((H, W, C), dtype=np.float32)
    valid_count = mask_stack.sum(axis=0)  # [H, W]

    # Build frequency radius map for sharpening (0 at DC, 1 at Nyquist)
    rfft_w = W // 2 + 1
    fy = np.fft.fftfreq(H)[:, None]  # [H, 1]
    fx = np.fft.rfftfreq(W)[None, :]  # [1, rfft_w]
    freq_radius = np.sqrt(fy ** 2 + fx ** 2)
    freq_radius = freq_radius / (freq_radius.max() + 1e-8)  # normalize to [0, 1]

    # Sharpening boost: ramp from 1.0 at cutoff to (1+strength) at Nyquist
    sharpen_boost = np.ones_like(freq_radius)
    above_cutoff = freq_radius > sharpen_cutoff
    sharpen_boost[above_cutoff] = 1.0 + sharpen_strength * (
        (freq_radius[above_cutoff] - sharpen_cutoff) / (1.0 - sharpen_cutoff + 1e-8)
    )

    for c in range(C):
        # FFT the center frame
        center_fft = np.fft.rfft2(center_f[:, :, c])

        # Collect FFTs from all aligned observations
        all_mags = []
        for i in range(N):
            obs = obs_stack[i, :, :, c].copy()
            mask = mask_stack[i]
            obs[~mask] = center_f[~mask, c]  # fill invalid with center
            obs_fft = np.fft.rfft2(obs)
            all_mags.append(np.abs(obs_fft))

        mag_stack_c = np.stack(all_mags, axis=0)  # [N, H, rfft_w]

        # Estimate per-frequency SNR from temporal statistics
        mag_mean = mag_stack_c.mean(axis=0)
        mag_std = mag_stack_c.std(axis=0)
        snr = mag_mean / (mag_std + 1e-8)

        # Wiener gain: SNR^2 / (SNR^2 + 1/strength)
        # strength > 1 = more aggressive denoising, < 1 = gentler
        wiener_gain = (snr ** 2) / (snr ** 2 + 1.0 / (wiener_strength + 1e-8))
        wiener_gain = np.clip(wiener_gain, 0.0, 1.0)

        # Step 1: Apply Wiener filter to CENTER FRAME (not average!)
        clean_fft = center_fft * wiener_gain

        # Step 2: Apply sharpening boost to high frequencies
        if sharpen_strength > 0:
            clean_fft = clean_fft * sharpen_boost

        # Reconstruct
        clean = np.fft.irfft2(clean_fft, s=(H, W))

        # Fall back to center where not enough observations
        fallback = valid_count < MIN_VALID_OBS
        clean[fallback] = center_f[fallback, c]

        result[:, :, c] = clean

    return np.clip(result, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Component 3: Phase correlation pre-alignment
# ---------------------------------------------------------------------------
def phase_correlate_shift(frame1_np, frame2_np):
    """Compute global translation between frames using FFT phase correlation.

    Near-instant computation that finds the dominant global shift between
    two frames. Used to pre-align before RAFT so RAFT only handles residual
    (non-translational) motion.

    Args:
        frame1_np: [H, W, 3] uint8 numpy array (reference frame)
        frame2_np: [H, W, 3] uint8 numpy array (frame to align)

    Returns:
        (dx, dy): pixel shift to apply to frame2 to align with frame1
    """
    # Convert to grayscale
    gray1 = cv2.cvtColor(frame1_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray2 = cv2.cvtColor(frame2_np, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Phase correlation
    F1 = np.fft.fft2(gray1)
    F2 = np.fft.fft2(gray2)
    cross_power = (F1 * np.conj(F2)) / (np.abs(F1 * np.conj(F2)) + 1e-8)
    correlation = np.abs(np.fft.ifft2(cross_power))

    # Find peak (= global shift)
    peak = np.unravel_index(np.argmax(correlation), correlation.shape)
    dy, dx = peak
    # Handle wraparound
    if dy > gray1.shape[0] // 2:
        dy -= gray1.shape[0]
    if dx > gray1.shape[1] // 2:
        dx -= gray1.shape[1]

    return float(dx), float(dy)


def apply_global_shift(frame_np, dx, dy):
    """Shift frame by (dx, dy) pixels using affine transform.

    Args:
        frame_np: [H, W, 3] uint8 numpy array
        dx: horizontal shift in pixels
        dy: vertical shift in pixels

    Returns:
        shifted: [H, W, 3] uint8 numpy array
    """
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(frame_np, M, (frame_np.shape[1], frame_np.shape[0]))


# ---------------------------------------------------------------------------
# Side-by-side comparison image
# ---------------------------------------------------------------------------
def make_comparison(original, raft_target, frame_idx, output_dir):
    """Save a side-by-side comparison: original | RAFT target."""
    H, W = original.shape[:2]
    # Add a 2px white separator
    sep = np.full((H, 2, 3), 255, dtype=np.uint8)
    combined = np.concatenate([original, sep, raft_target], axis=1)
    path = os.path.join(output_dir, f"compare_{frame_idx:06d}.png")
    Image.fromarray(combined).save(path)
    return path


# ---------------------------------------------------------------------------
# Progress / resume tracking
# ---------------------------------------------------------------------------
def load_progress(output_dir):
    """Load set of already-processed frame indices."""
    progress_file = os.path.join(output_dir, "_progress.json")
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            data = json.load(f)
        return set(data.get("completed", []))
    return set()


def save_progress(output_dir, completed_set):
    """Save set of completed frame indices for resume."""
    progress_file = os.path.join(output_dir, "_progress.json")
    with open(progress_file, "w") as f:
        json.dump({"completed": sorted(completed_set)}, f)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def process_video(
    input_path,
    output_dir,
    window=4,
    max_frames=None,
    save_comparison=False,
    small=None,
    recursive=False,
    prefilter=False,
    prefilter_sigma=2.0,
    fft_denoise=False,
    phase_align=False,
    wiener_sharpen=False,
    wiener_strength_val=1.0,
    sharpen_strength_val=1.0,
):
    """Generate temporally-averaged targets for a video clip.

    Args:
        input_path: path to input video file
        output_dir: directory to save output PNGs
        window: number of frames on each side (total = 2*window + 1)
        max_frames: limit total frames processed (for testing)
        save_comparison: save side-by-side comparisons
        small: use RAFT-Small model (default: USE_SMALL)
        recursive: use already-cleaned frames as neighbors
        prefilter: low-pass filter RAFT inputs for cleaner flow
        prefilter_sigma: Gaussian sigma for pre-filter
        fft_denoise: use FFT confidence-weighted averaging instead of median
        phase_align: pre-align frames via FFT phase correlation before RAFT
    """
    # Setup output directories
    targets_dir = os.path.join(output_dir, "targets")
    os.makedirs(targets_dir, exist_ok=True)
    if save_comparison:
        compare_dir = os.path.join(output_dir, "comparisons")
        os.makedirs(compare_dir, exist_ok=True)

    # Load progress for resume
    completed = load_progress(output_dir)
    if completed:
        print(f"[resume] Found {len(completed)} already-completed frames", flush=True)

    # Extract all frames to RAM
    print(f"[main] Extracting frames from: {input_path}", flush=True)
    frames, w, h, fps = extract_frames_to_numpy(input_path, max_frames)
    num_frames = len(frames)

    if num_frames == 0:
        print("[ERROR] No frames extracted!", flush=True)
        return

    ram_gb = num_frames * w * h * 3 / 1e9
    print(f"[main] {num_frames} frames loaded, {ram_gb:.1f} GB RAM", flush=True)
    print(f"[main] Window size: {window} (using {2 * window + 1} frames per target)", flush=True)

    # Load RAFT model
    model = load_raft_model(small=small)

    # For recursive mode: store cleaned frames to use as neighbors
    cleaned_frames = {}
    if recursive:
        print("[main] Recursive mode: using cleaned frames as neighbors", flush=True)

    # Determine if we can use GPU-accelerated path
    use_gpu_accel = torch.cuda.is_available() and not fft_denoise
    if use_gpu_accel:
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        # Estimate max batch size for flow computation
        # At 1080p, RAFT correlation volume uses ~4GB per pair
        # (fmap at 1/8 res: 135x240, all-pairs corr = 135*240*135*240 floats)
        # Leave 10GB headroom for model + warped frame storage
        est_per_pair_gb = (h * w) / (1920 * 1080) * 4.0  # scale from 1080p baseline
        flow_batch_size = max(1, int((gpu_mem_gb - 10) / max(est_per_pair_gb, 0.5)))
        flow_batch_size = min(flow_batch_size, 8)  # cap at 8
        print(
            f"[main] GPU-accelerated path: batch_size={flow_batch_size}, "
            f"GPU median, GPU warp",
            flush=True,
        )
    else:
        flow_batch_size = 1
        if fft_denoise:
            print("[main] FFT denoise mode: using CPU path", flush=True)

    # Try torch.compile for faster RAFT inference
    compiled_model = model
    if use_gpu_accel:
        try:
            compiled_model = torch.compile(model, mode="reduce-overhead")
            # Warm up with a dummy forward pass
            dummy1 = torch.randn(1, 3, 64, 64, device=DEVICE)
            dummy2 = torch.randn(1, 3, 64, 64, device=DEVICE)
            with torch.no_grad():
                compiled_model(dummy1, dummy2, iters=4, test_mode=True)
            del dummy1, dummy2
            torch.cuda.empty_cache()
            print("[main] torch.compile: OK (reduce-overhead)", flush=True)
        except Exception as e:
            compiled_model = model
            print(f"[main] torch.compile failed, using eager: {e}", flush=True)

    # Process each frame
    t_start = time.time()
    flow_times = []
    for t in range(num_frames):
        if t in completed:
            # Load already-completed target for recursive mode
            if recursive:
                target_path = os.path.join(targets_dir, f"target_{t:06d}.png")
                if os.path.exists(target_path):
                    loaded = np.array(Image.open(target_path))
                    cleaned_frames[t] = loaded
            continue

        t_frame_start = time.time()

        # Determine neighbor range (clamp to valid indices)
        n_start = max(0, t - window)
        n_end = min(num_frames - 1, t + window)
        neighbors = [n for n in range(n_start, n_end + 1) if n != t]

        center_frame = frames[t]

        # Collect neighbor frames (with recursive substitution and phase align)
        neighbor_frames = []
        for n in neighbors:
            if recursive and n in cleaned_frames:
                nf = cleaned_frames[n]
            else:
                nf = frames[n]
            if phase_align:
                dx, dy = phase_correlate_shift(center_frame, nf)
                if abs(dx) > 0.5 or abs(dy) > 0.5:
                    nf = apply_global_shift(nf, dx, dy)
            neighbor_frames.append(nf)

        # Optionally pre-filter for cleaner flow
        if prefilter:
            center_flow_input = prefilter_for_flow(center_frame, sigma=prefilter_sigma)
            neighbor_flow_inputs = [
                prefilter_for_flow(nf, sigma=prefilter_sigma) for nf in neighbor_frames
            ]
        else:
            center_flow_input = center_frame
            neighbor_flow_inputs = neighbor_frames

        t_flow_start = time.time()

        if use_gpu_accel and flow_batch_size > 1:
            # -- Batched GPU flow computation --
            # Build pairs: [fwd_0, bwd_0, fwd_1, bwd_1, ...]
            all_pairs = []
            for nfi in neighbor_flow_inputs:
                all_pairs.append((center_flow_input, nfi))   # forward
                all_pairs.append((nfi, center_flow_input))   # backward
            all_flows = compute_flow_batch(
                compiled_model, all_pairs, max_batch=flow_batch_size
            )
            # all_flows is on GPU: [fwd_0, bwd_0, fwd_1, bwd_1, ...]

            t_flow_end = time.time()
            flow_times.append(t_flow_end - t_flow_start)

            # Warp and occlusion on GPU
            warped_gpu_list = []
            mask_gpu_list = []
            for ni in range(len(neighbors)):
                flow_fwd_gpu = all_flows[ni * 2]      # [1, 2, H, W] GPU
                flow_bwd_gpu = all_flows[ni * 2 + 1]  # [1, 2, H, W] GPU

                warped_gpu, in_bounds_gpu = warp_frame_gpu(
                    neighbor_frames[ni], flow_fwd_gpu
                )
                valid_flow_gpu = compute_occlusion_mask_gpu(flow_fwd_gpu, flow_bwd_gpu)
                combined_mask_gpu = in_bounds_gpu & valid_flow_gpu

                warped_gpu_list.append(warped_gpu)
                mask_gpu_list.append(combined_mask_gpu)

            del all_flows

            # GPU median
            t_median_start = time.time()
            target = compute_temporal_median_gpu(
                center_frame, warped_gpu_list, mask_gpu_list
            )
            t_median = time.time() - t_median_start

            del warped_gpu_list, mask_gpu_list
            torch.cuda.empty_cache()
        else:
            # -- Original sequential CPU path (for FFT denoise or small GPU) --
            warped_list = []
            mask_list = []
            for ni, n in enumerate(neighbors):
                nfi = neighbor_flow_inputs[ni]
                nf = neighbor_frames[ni]

                t_flow_pair_start = time.time()
                flow_fwd = compute_flow_pair(
                    compiled_model, center_flow_input, nfi,
                )
                flow_bwd = compute_flow_pair(
                    compiled_model, nfi, center_flow_input,
                )
                t_flow_pair_end = time.time()
                flow_times.append(t_flow_pair_end - t_flow_pair_start)

                warped, in_bounds = warp_frame(nf, flow_fwd)
                valid_flow = compute_occlusion_mask(flow_fwd, flow_bwd)
                combined_mask = in_bounds & valid_flow

                warped_list.append(warped)
                mask_list.append(combined_mask)
                del flow_fwd, flow_bwd

                if len(neighbors) > 4:
                    print(
                        f"    neighbor {ni + 1}/{len(neighbors)} "
                        f"(t={t} n={n}) "
                        f"{t_flow_pair_end - t_flow_pair_start:.1f}s | "
                        f"{gpu_mem_str()}",
                        flush=True,
                    )

            t_median_start = time.time()
            if wiener_sharpen:
                target = compute_wiener_sharpen(
                    center_frame, warped_list, mask_list,
                    wiener_strength=wiener_strength_val,
                    sharpen_strength=sharpen_strength_val,
                )
            elif fft_denoise:
                target = compute_temporal_fft_confidence(
                    center_frame, warped_list, mask_list
                )
            else:
                target = compute_temporal_median(center_frame, warped_list, mask_list)
            t_median = time.time() - t_median_start
            del warped_list, mask_list

        # Save target
        target_path = os.path.join(targets_dir, f"target_{t:06d}.png")
        Image.fromarray(target).save(target_path)

        # Store for recursive mode (future frames use this as neighbor)
        if recursive:
            cleaned_frames[t] = target
            # Only keep frames within window range to limit RAM
            stale = [k for k in cleaned_frames if k < t - window]
            for k in stale:
                del cleaned_frames[k]

        # Save comparison if requested
        if save_comparison:
            cmp_path = make_comparison(center_frame, target, t, compare_dir)

        # Update progress
        completed.add(t)
        if (t + 1) % 5 == 0 or t == num_frames - 1:
            save_progress(output_dir, completed)

        # Stats
        elapsed = time.time() - t_frame_start
        total_elapsed = time.time() - t_start
        frames_done = len(completed)
        frames_remaining = num_frames - frames_done
        avg_time = total_elapsed / max(frames_done, 1)
        eta_sec = frames_remaining * avg_time
        avg_flow_time = flow_times[-1] if flow_times else 0

        print(
            f"  frame {t + 1}/{num_frames} | "
            f"{len(neighbors)} neighbors | "
            f"{elapsed:.1f}s (flow {avg_flow_time:.2f}s, median {t_median:.2f}s) | "
            f"ETA: {eta_sec / 60:.1f}min | "
            f"{gpu_mem_str()}",
            flush=True,
        )

        # Clean up per-frame temporaries
        gc.collect()

    total_time = time.time() - t_start
    print(f"\n[done] Processed {len(completed)} frames in {total_time / 60:.1f} min", flush=True)
    print(f"[done] Targets saved to: {targets_dir}", flush=True)
    if save_comparison:
        print(f"[done] Comparisons saved to: {compare_dir}", flush=True)

    # Save final progress
    save_progress(output_dir, completed)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    global RAFT_ITERS, FB_CONSISTENCY_THRESH, USE_SMALL

    parser = argparse.ArgumentParser(
        description="Generate temporally-averaged targets using RAFT optical flow"
    )
    parser.add_argument("-i", "--input", required=True, help="Input video file (mkv/mp4)")
    parser.add_argument("-o", "--output", required=True, help="Output directory for PNGs")
    parser.add_argument(
        "--window", type=int, default=4,
        help="Number of frames on each side (default: 4 = 9 total)"
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Limit total frames to process (for testing)"
    )
    parser.add_argument(
        "--comparison", action="store_true",
        help="Save side-by-side comparison PNGs (original | RAFT target)"
    )
    parser.add_argument(
        "--small", action="store_true", default=True,
        help="Use RAFT-Small model (default: True, less VRAM)"
    )
    parser.add_argument(
        "--no-small", action="store_true",
        help="Use full RAFT-Things model instead of RAFT-Small"
    )
    parser.add_argument(
        "--raft-iters", type=int, default=RAFT_ITERS,
        help=f"RAFT iterations (default: {RAFT_ITERS})"
    )
    parser.add_argument(
        "--fb-threshold", type=float, default=FB_CONSISTENCY_THRESH,
        help=f"Forward-backward consistency threshold in pixels (default: {FB_CONSISTENCY_THRESH})"
    )
    parser.add_argument(
        "--half-res", action="store_true",
        help="Compute flow at half resolution (saves VRAM, slightly less accurate)"
    )
    parser.add_argument(
        "--fft", action="store_true",
        help="Use simple FFT magnitude averaging instead of spatial median"
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="Use already-cleaned frames as neighbors (quality compounds, must be sequential)"
    )
    parser.add_argument(
        "--wiener-sharpen", action="store_true",
        help="Wiener filter center frame using temporal stats, then sharpen (best quality)"
    )
    parser.add_argument(
        "--wiener-strength", type=float, default=1.0,
        help="Wiener denoising strength (0.5=gentle, 1.0=standard, 2.0=aggressive)"
    )
    parser.add_argument(
        "--sharpen-strength", type=float, default=1.0,
        help="High-frequency boost strength (0=none, 1.0=moderate, 2.0=strong)"
    )
    parser.add_argument(
        "--prefilter", action="store_true",
        help="Gaussian pre-filter RAFT inputs for cleaner flow (default sigma=2.0)"
    )
    parser.add_argument(
        "--prefilter-sigma", type=float, default=2.0,
        help="Sigma for pre-filter Gaussian blur (default: 2.0)"
    )
    parser.add_argument(
        "--fft-denoise", action="store_true",
        help="Use FFT confidence-weighted averaging instead of spatial median"
    )
    parser.add_argument(
        "--phase-align", action="store_true",
        help="Pre-align frames via FFT phase correlation before RAFT"
    )

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.input):
        print(f"[ERROR] Input file not found: {args.input}", flush=True)
        sys.exit(1)

    # Override module-level defaults with CLI args
    RAFT_ITERS = args.raft_iters
    FB_CONSISTENCY_THRESH = args.fb_threshold
    global FLOW_HALF_RES
    FLOW_HALF_RES = args.half_res
    use_small = args.small and not args.no_small
    USE_SMALL = use_small

    model_name = "RAFT-Small" if use_small else "RAFT-Things"
    model_path = RAFT_SMALL_PATH if use_small else RAFT_THINGS_PATH

    print("=" * 60, flush=True)
    print("RAFT Temporal Target Generator", flush=True)
    print("=" * 60, flush=True)
    print(f"  Input:       {args.input}", flush=True)
    print(f"  Output:      {args.output}", flush=True)
    print(f"  Window:      {args.window} ({2 * args.window + 1} frames)", flush=True)
    print(f"  Max frames:  {args.max_frames or 'all'}", flush=True)
    print(f"  Comparison:  {args.comparison}", flush=True)
    print(f"  RAFT model:  {model_name} ({model_path})", flush=True)
    print(f"  RAFT iters:  {args.raft_iters}", flush=True)
    print(f"  FB thresh:   {args.fb_threshold} px", flush=True)
    print(f"  Prefilter:   {args.prefilter} (sigma={args.prefilter_sigma})", flush=True)
    print(f"  FFT denoise: {args.fft_denoise}", flush=True)
    print(f"  Phase align: {args.phase_align}", flush=True)
    print(f"  Device:      {DEVICE}", flush=True)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU:         {gpu_name} ({gpu_mem:.1f} GB)", flush=True)
    print("=" * 60, flush=True)

    process_video(
        input_path=args.input,
        output_dir=args.output,
        window=args.window,
        max_frames=args.max_frames,
        save_comparison=args.comparison,
        small=use_small,
        recursive=args.recursive,
        prefilter=args.prefilter,
        prefilter_sigma=args.prefilter_sigma,
        fft_denoise=args.fft_denoise,
        phase_align=args.phase_align,
        wiener_sharpen=args.wiener_sharpen,
        wiener_strength_val=args.wiener_strength,
        sharpen_strength_val=args.sharpen_strength,
    )


if __name__ == "__main__":
    main()
