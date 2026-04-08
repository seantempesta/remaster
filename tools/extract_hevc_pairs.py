"""
Extract HEVC training pairs from Firefly episodes using the pretrained DRUNet denoiser.

For each randomly sampled frame:
- The compressed source frame is saved as INPUT
- The denoised frame (drunet_color with configurable sigma) is saved as TARGET

Uses the pretrained drunet_color.pth Gaussian denoiser with a noise level map,
which handles both compression artifacts and film grain in one pass.

Usage:
    python tools/extract_hevc_pairs.py \
        --output-dir data/mixed_pairs --val-dir data/mixed_val \
        --num-frames 1200 --num-val 120 --sigma 10
"""
import sys
import os
import gc
import re
import random
import subprocess
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.ffmpeg_utils import get_ffmpeg
from lib.paths import resolve_kair_dir

FFMPEG = get_ffmpeg()

PLEX_DIR = r"E:\plex\tv\Firefly (2002) Season 1 S01 (1080p BluRay x265 HEVC 10bit AAC Silence)"

EPISODES = {
    "e01": "Firefly (2002) - S01E01 - Serenity (1080p x265 Silence).mkv",
    "e02": "Firefly (2002) - S01E02 - The Train Job (1080p BluRay x265 Silence).mkv",
    "e03": "Firefly (2002) - S01E03 - Bushwhacked (1080p BluRay x265 Silence).mkv",
    "e04": "Firefly (2002) - S01E04 - Shindig (1080p BluRay x265 Silence).mkv",
    "e05": "Firefly (2002) - S01E05 - Safe (1080p BluRay x265 Silence).mkv",
    "e06": "Firefly (2002) - S01E06 - Our Mrs. Reynolds (1080p BluRay x265 Silence).mkv",
    "e07": "Firefly (2002) - S01E07 - Jaynestown (1080p BluRay x265 Silence).mkv",
    "e08": "Firefly (2002) - S01E08 - Out of Gas (1080p BluRay x265 Silence).mkv",
    "e09": "Firefly (2002) - S01E09 - Ariel (1080p BluRay x265 Silence).mkv",
    "e10": "Firefly (2002) - S01E10 - War Stories (1080p BluRay x265 Silence).mkv",
    "e11": "Firefly (2002) - S01E11 - Trash (1080p BluRay x265 Silence).mkv",
    "e12": "Firefly (2002) - S01E12 - The Message (1080p BluRay x265 Silence).mkv",
    "e13": "Firefly (2002) - S01E13 - Heart of Gold (1080p BluRay x265 Silence).mkv",
    "e14": "Firefly (2002) - S01E14 - Objects in Space (1080p BluRay x265 Silence).mkv",
}


def probe_video(video_path):
    """Probe video for width, height, codec, duration."""
    r = subprocess.run(
        [FFMPEG, "-hide_banner", "-i", video_path],
        capture_output=True, text=True,
    )
    stderr = r.stderr

    video_match = re.search(r'Video:\s+(\w+).*?(\d{3,5})x(\d{3,5})', stderr)
    if not video_match:
        raise ValueError(f"Could not detect video stream in {video_path}")

    codec = video_match.group(1).lower()
    width = int(video_match.group(2))
    height = int(video_match.group(3))

    dur_match = re.search(r'Duration:\s+(\d+):(\d+):(\d+\.\d+)', stderr)
    if dur_match:
        duration = (int(dur_match.group(1)) * 3600
                    + int(dur_match.group(2)) * 60
                    + float(dur_match.group(3)))
    else:
        raise ValueError(f"Could not detect duration in {video_path}")

    return width, height, codec, duration


def extract_frame(video_path, timestamp_sec, width, height, codec):
    """Extract a single frame using NVDEC hwaccel."""
    CUVID = {"h264": "h264_cuvid", "hevc": "hevc_cuvid"}
    expected_bytes = width * height * 3

    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error"]
    cuvid = CUVID.get(codec)
    if cuvid:
        cmd += ["-hwaccel", "cuda", "-c:v", cuvid]
    cmd += [
        "-ss", f"{timestamp_sec:.3f}",
        "-i", video_path,
        "-frames:v", "1",
        "-f", "image2pipe",
        "-pix_fmt", "bgr24",
        "-vcodec", "rawvideo",
        "pipe:1",
    ]
    r = subprocess.run(cmd, capture_output=True)

    if r.returncode != 0 or len(r.stdout) != expected_bytes:
        # Fallback to software decode
        cmd = [
            FFMPEG, "-hide_banner", "-loglevel", "error",
            "-ss", f"{timestamp_sec:.3f}",
            "-i", video_path,
            "-frames:v", "1",
            "-f", "image2pipe",
            "-pix_fmt", "bgr24",
            "-vcodec", "rawvideo",
            "pipe:1",
        ]
        r = subprocess.run(cmd, capture_output=True)

    if r.returncode != 0 or len(r.stdout) != expected_bytes:
        return None

    return np.frombuffer(r.stdout, dtype=np.uint8).reshape(height, width, 3)


def load_denoiser(device="cuda"):
    """Load pretrained DRUNet Gaussian denoiser (4-channel: RGB + noise level map)."""
    kair_dir = resolve_kair_dir()
    sys.path.insert(0, str(kair_dir))
    from models.network_unet import UNetRes

    weights_path = os.path.join(str(kair_dir), "model_zoo", "drunet_color.pth")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Pretrained denoiser not found at {weights_path}. "
            "Download from https://github.com/cszn/KAIR/releases/download/v1.0/drunet_color.pth"
        )

    model = UNetRes(
        in_nc=4, out_nc=3,
        nc=[64, 128, 256, 512], nb=4,
        act_mode='R', bias=False,
    )

    ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt, strict=True)
    del ckpt
    gc.collect()

    model = model.half().to(device).eval()
    print(f"  Denoiser loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params, "
          f"VRAM: {torch.cuda.memory_allocated()/1024**2:.0f}MB")
    return model


@torch.no_grad()
def denoise_frame(model, frame_bgr, sigma, device="cuda"):
    """Denoise a BGR uint8 frame using DRUNet with noise level map.

    Args:
        model: DRUNet model with 4-channel input (RGB + noise level)
        frame_bgr: BGR uint8 numpy array
        sigma: noise level (0-255 scale, e.g. 10 for light denoising)
        device: torch device
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    tensor = tensor.half().to(device)

    # Add noise level map as 4th channel
    _, _, h, w = tensor.shape
    noise_map = torch.full((1, 1, h, w), sigma / 255.0, dtype=tensor.dtype, device=device)
    tensor = torch.cat([tensor, noise_map], dim=1)

    # Pad to multiple of 8 (UNet requirement)
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
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    return output


def generate_hevc_pairs(episodes, output_dir, num_frames, denoiser, sigma,
                        seed=42, skip_existing=True):
    """Extract frames and generate denoised target pairs."""
    if num_frames <= 0:
        return 0

    input_dir = os.path.join(output_dir, "input")
    target_dir = os.path.join(output_dir, "target")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)

    num_episodes = len(episodes)
    frames_per_ep = num_frames // num_episodes
    remainder = num_frames % num_episodes

    total_generated = 0
    total_skipped = 0
    t0 = time.time()

    for i, (tag, filename) in enumerate(sorted(episodes.items())):
        video_path = os.path.join(PLEX_DIR, filename)
        if not os.path.exists(video_path):
            print(f"  SKIP: {tag} not found")
            continue

        width, height, codec, duration = probe_video(video_path)
        n_frames = frames_per_ep + (1 if i < remainder else 0)

        # Skip first/last 90 seconds
        start_t = 90
        end_t = duration - 90
        if end_t <= start_t:
            start_t, end_t = 30, duration - 30

        ep_rng = random.Random(seed + hash(tag))
        timestamps = sorted(ep_rng.uniform(start_t, end_t) for _ in range(n_frames))

        ep_generated = 0
        ep_skipped = 0

        for j, ts in enumerate(timestamps):
            filename_png = f"hevc_{tag}_{j:05d}.png"
            input_path = os.path.join(input_dir, filename_png)
            target_path = os.path.join(target_dir, filename_png)

            if skip_existing and os.path.exists(input_path) and os.path.exists(target_path):
                ep_skipped += 1
                continue

            frame = extract_frame(video_path, ts, width, height, codec)
            if frame is None:
                print(f"    WARN: failed to extract {tag} at {ts:.1f}s")
                continue

            target = denoise_frame(denoiser, frame, sigma)

            cv2.imwrite(input_path, frame)
            cv2.imwrite(target_path, target)
            ep_generated += 1

        total_generated += ep_generated
        total_skipped += ep_skipped
        status = f"{ep_generated} new"
        if ep_skipped:
            status += f", {ep_skipped} skipped"
        elapsed = time.time() - t0
        fps = total_generated / elapsed if elapsed > 0 else 0
        print(f"  {tag}: {width}x{height} {codec} - {status} ({fps:.1f} fps avg)")

    return total_generated


def main():
    parser = argparse.ArgumentParser(
        description="Extract HEVC training pairs using DRUNet teacher")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for training pairs")
    parser.add_argument("--val-dir", type=str, default=None,
                        help="Output directory for validation pairs")
    parser.add_argument("--sigma", type=int, default=10,
                        help="Denoiser noise level (default: 10)")
    parser.add_argument("--num-frames", type=int, default=1200,
                        help="Total training frames (default: 1200)")
    parser.add_argument("--num-val", type=int, default=120,
                        help="Total validation frames (default: 120)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--no-skip", action="store_true",
                        help="Regenerate all frames")
    args = parser.parse_args()

    print(f"Loading denoiser (sigma={args.sigma})...")
    denoiser = load_denoiser()

    if args.num_frames > 0:
        print(f"\n=== TRAINING SET: {args.num_frames} frames across {len(EPISODES)} episodes ===")
        print(f"Output: {args.output_dir}")
        n_train = generate_hevc_pairs(
            EPISODES, args.output_dir, args.num_frames, denoiser, args.sigma,
            seed=args.seed, skip_existing=not args.no_skip,
        )
        print(f"\nTraining: {n_train} pairs")
    else:
        n_train = 0

    if args.val_dir and args.num_val > 0:
        print(f"\n=== VALIDATION SET: {args.num_val} frames ===")
        print(f"Output: {args.val_dir}")
        n_val = generate_hevc_pairs(
            EPISODES, args.val_dir, args.num_val, denoiser, args.sigma,
            seed=args.seed + 9999, skip_existing=not args.no_skip,
        )
        print(f"Validation: {n_val} pairs")
    else:
        n_val = 0

    print(f"\nDone: {n_train} train + {n_val} val pairs")

    del denoiser
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
