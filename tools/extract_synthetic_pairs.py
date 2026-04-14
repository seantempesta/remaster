"""
Extract synthetic training pairs from high-quality video source material.

For each extracted frame:
- The original is saved as the TARGET (sharp ground truth)
- An edge-aware degraded version is saved as the INPUT

Edge-aware degradation:
- Computes edge magnitude map (Sobel)
- Sharp areas (edges, texture, faces) get blurred MORE
- Already-soft areas (bokeh, sky, gradients) stay mostly untouched
- Randomized per-frame sigma for robustness
- Optional mild downscale+upscale on a fraction of frames

Usage:
    # Extract from a directory of episodes
    python tools/extract_synthetic_pairs.py \\
        --source-dir "/path/to/Show Name" --pattern "*.mkv" \\
        --name synth_showname \\
        --output-dir data/mixed_pairs --val-dir data/mixed_val \\
        --num-frames 400 --num-val 40

    # Extract from explicit file list (loose files or single movie)
    python tools/extract_synthetic_pairs.py \\
        --files movie.mp4 \\
        --name synth_movie --tag-override movie \\
        --output-dir data/mixed_pairs --val-dir data/mixed_val \\
        --num-frames 300 --num-val 30

    # Test mode (10 pairs from first 2 episodes)
    python tools/extract_synthetic_pairs.py \\
        --source-dir "/path/to/Show" --pattern "*.mkv" \\
        --name synth_show --output-dir data/test_pairs --test
"""
import sys
import os
import re
import json
import random
import subprocess
import argparse
import glob
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.ffmpeg_utils import get_ffmpeg

FFMPEG = get_ffmpeg()

# NVDEC cuvid decoders by codec name
CUVID_DECODERS = {
    "h264": "h264_cuvid",
    "hevc": "hevc_cuvid",
    "vp9": "vp9_cuvid",
    "av1": "av1_cuvid",
}


@dataclass
class VideoInfo:
    """Metadata for a video file, probed once and reused."""
    path: str
    width: int
    height: int
    codec: str
    duration: float
    cuvid_decoder: str | None


def probe_video(video_path):
    """Probe a video file for all metadata needed by the extraction pipeline.

    Uses ffmpeg to extract width, height, codec, and duration in a single call.
    Returns a VideoInfo dataclass. Raises ValueError if probing fails.
    """
    # ffmpeg -i prints stream info to stderr; we parse it once for everything
    r = subprocess.run(
        [FFMPEG, "-hide_banner", "-i", video_path],
        capture_output=True, text=True,
    )
    stderr = r.stderr

    # Parse video stream: "Video: h264 (Main), yuv420p, 1920x1080"
    video_match = re.search(
        r'Video:\s+(\w+).*?(\d{3,5})x(\d{3,5})', stderr
    )
    if not video_match:
        raise ValueError(f"Could not detect video stream in {video_path}")

    codec = video_match.group(1).lower()
    width = int(video_match.group(2))
    height = int(video_match.group(3))

    # Parse duration: "Duration: 00:42:31.12"
    dur_match = re.search(r'Duration:\s+(\d+):(\d+):(\d+\.\d+)', stderr)
    if dur_match:
        duration = (int(dur_match.group(1)) * 3600
                    + int(dur_match.group(2)) * 60
                    + float(dur_match.group(3)))
    else:
        raise ValueError(f"Could not detect duration in {video_path}")

    cuvid = CUVID_DECODERS.get(codec)

    return VideoInfo(
        path=video_path,
        width=width,
        height=height,
        codec=codec,
        duration=duration,
        cuvid_decoder=cuvid,
    )


def extract_frame(video_info, timestamp_sec):
    """Extract a single frame at a timestamp, return as numpy array (BGR).

    Uses NVDEC hardware decoding when available, with software fallback.
    Resolution is known from probe_video, so no guessing needed.
    """
    expected_bytes = video_info.width * video_info.height * 3

    def _run_extract(use_hwaccel):
        cmd = [FFMPEG, "-hide_banner", "-loglevel", "error"]
        if use_hwaccel and video_info.cuvid_decoder:
            cmd += ["-hwaccel", "cuda", "-c:v", video_info.cuvid_decoder]
        cmd += [
            "-ss", f"{timestamp_sec:.3f}",
            "-i", video_info.path,
            "-frames:v", "1",
            "-f", "image2pipe",
            "-pix_fmt", "bgr24",
            "-vcodec", "rawvideo",
            "pipe:1",
        ]
        return subprocess.run(cmd, capture_output=True)

    # Try hardware decode first
    r = _run_extract(use_hwaccel=True)

    # Fallback to software if hwaccel failed
    if (r.returncode != 0 or len(r.stdout) != expected_bytes) and video_info.cuvid_decoder:
        r = _run_extract(use_hwaccel=False)

    if r.returncode != 0 or len(r.stdout) != expected_bytes:
        return None

    return np.frombuffer(r.stdout, dtype=np.uint8).reshape(
        video_info.height, video_info.width, 3
    )


def edge_aware_degrade(image, base_sigma, rng, downscale_prob=0.15):
    """Apply edge-aware degradation to an image.

    Sharp areas (high edge magnitude) get blurred more.
    Soft areas (low edge magnitude) stay mostly untouched.
    """
    h, w = image.shape[:2]
    img_f = image.astype(np.float32) / 255.0

    # Compute edge map
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    edge_mag = edge_mag / (edge_mag.max() + 1e-8)

    # Smooth the edge map so blur transitions are gradual
    edge_map = cv2.GaussianBlur(edge_mag, (0, 0), sigmaX=5.0)
    edge_map = edge_map / (edge_map.max() + 1e-8)

    # Spatially-varying blur: blend original with strongly-blurred version
    max_sigma = base_sigma * 2.0
    ksize = int(max_sigma * 6) | 1  # ensure odd
    ksize = max(ksize, 3)
    blurred = cv2.GaussianBlur(img_f, (ksize, ksize), sigmaX=max_sigma)

    # Edge map controls blend: high edge = more blur, low edge = keep original
    alpha = edge_map[:, :, np.newaxis]
    alpha = alpha * min(base_sigma / 2.0, 1.0)
    degraded = img_f * (1.0 - alpha) + blurred * alpha

    # Optional downscale + upscale (simulates resolution loss)
    if rng.random() < downscale_prob:
        scale = rng.uniform(0.5, 0.75)
        small_h, small_w = int(h * scale), int(w * scale)
        degraded = cv2.resize(degraded, (small_w, small_h), interpolation=cv2.INTER_AREA)
        degraded = cv2.resize(degraded, (w, h), interpolation=cv2.INTER_CUBIC)

    # Add light Gaussian noise (sigma 1-5 in [0,255] range -> 0.004-0.02 in [0,1])
    # Teaches model to suppress noise AND recover detail simultaneously
    noise_sigma = rng.uniform(1.0, 5.0) / 255.0
    np_rng = np.random.default_rng(rng.getrandbits(64))
    noise = np_rng.standard_normal(degraded.shape, dtype=np.float32) * noise_sigma
    degraded = degraded + noise

    degraded = np.clip(degraded * 255.0, 0, 255).astype(np.uint8)
    return degraded


def discover_episodes(source_dir, pattern="*.mkv"):
    """Auto-discover video files matching a glob pattern.

    Returns:
        dict of {tag: full_path} where tag is SxxExx or cleaned filename stem.
    """
    files = sorted(glob.glob(os.path.join(glob.escape(source_dir), pattern)))
    episodes = {}
    for f in files:
        basename = os.path.basename(f)
        m = re.search(r'S(\d+)E(\d+)', basename, re.IGNORECASE)
        if m:
            tag = f"S{m.group(1).zfill(2)}E{m.group(2).zfill(2)}"
        else:
            tag = Path(basename).stem[:20].replace(" ", "_")
        episodes[tag] = os.path.abspath(f)
    return episodes


def episodes_from_files(file_paths):
    """Build episodes dict from explicit file paths.

    Returns:
        dict of {tag: full_path} where tag is SxxExx or cleaned filename stem.
    """
    episodes = {}
    for f in file_paths:
        f = os.path.abspath(f)
        basename = os.path.basename(f)
        m = re.search(r'S(\d+)E(\d+)', basename, re.IGNORECASE)
        if m:
            tag = f"S{m.group(1).zfill(2)}E{m.group(2).zfill(2)}"
        else:
            tag = Path(basename).stem[:20].replace(" ", "_")
        episodes[tag] = f
    return episodes


def generate_pairs(episodes, name, output_dir, num_frames,
                   seed=42, sigma_range=(2.0, 5.0), downscale_prob=0.15,
                   skip_existing=True, tag_override=None):
    """Extract frames and generate degraded pairs.

    Probes each video file once for metadata (resolution, codec, duration),
    then uses NVDEC hardware decoding for frame extraction.
    """
    if num_frames <= 0:
        return 0

    input_dir = os.path.join(output_dir, "input")
    target_dir = os.path.join(output_dir, "target")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)

    rng = random.Random(seed)
    num_episodes = len(episodes)
    frames_per_ep = num_frames // num_episodes
    remainder = num_frames % num_episodes

    total_generated = 0
    total_skipped = 0

    for i, (tag, video_path) in enumerate(sorted(episodes.items())):
        if not os.path.exists(video_path):
            print(f"  SKIP: {tag} not found at {video_path}")
            continue

        # Probe video once for all metadata
        try:
            info = probe_video(video_path)
        except ValueError as e:
            print(f"  SKIP: {tag} - {e}")
            continue

        hwaccel_str = f", NVDEC {info.cuvid_decoder}" if info.cuvid_decoder else ", software decode"
        print(f"  {tag}: {info.width}x{info.height} {info.codec}{hwaccel_str}")

        n_frames = frames_per_ep + (1 if i < remainder else 0)

        # Skip first/last 90 seconds (credits, black frames)
        start_t = 90
        end_t = info.duration - 90
        if end_t <= start_t:
            start_t, end_t = 30, info.duration - 30
        if end_t <= start_t:
            start_t, end_t = 0, info.duration

        ep_rng = random.Random(seed + hash(tag))
        timestamps = sorted(ep_rng.uniform(start_t, end_t) for _ in range(n_frames))

        frame_tag_base = tag_override if tag_override else tag
        ep_generated = 0
        ep_skipped = 0

        for j, ts in enumerate(timestamps):
            filename = f"{name}_{frame_tag_base}_{j:05d}.png"
            input_path = os.path.join(input_dir, filename)
            target_path = os.path.join(target_dir, filename)

            if skip_existing and os.path.exists(input_path) and os.path.exists(target_path):
                ep_skipped += 1
                continue

            frame = extract_frame(info, ts)
            if frame is None:
                print(f"    WARN: failed to extract {tag} at {ts:.1f}s")
                continue

            base_sigma = rng.uniform(*sigma_range)
            degraded = edge_aware_degrade(frame, base_sigma, rng,
                                          downscale_prob=downscale_prob)

            cv2.imwrite(target_path, frame)
            cv2.imwrite(input_path, degraded)
            ep_generated += 1

        total_generated += ep_generated
        total_skipped += ep_skipped
        status = f"{ep_generated} new"
        if ep_skipped:
            status += f", {ep_skipped} skipped"
        print(f"    -> {status} (sigma {sigma_range[0]:.1f}-{sigma_range[1]:.1f})")

    return total_generated


def main():
    parser = argparse.ArgumentParser(
        description="Extract synthetic training pairs with edge-aware degradation")

    # Source selection (one of these is required)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-dir", type=str,
                        help="Directory containing video files (use with --pattern)")
    source.add_argument("--files", nargs="+", type=str,
                        help="Explicit list of video file paths")

    # Required
    parser.add_argument("--name", type=str, required=True,
                        help="Source name for output filenames (e.g. 'synth_onepiece')")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for training pairs (creates input/ and target/)")

    # Optional
    parser.add_argument("--pattern", type=str, default="*.mkv",
                        help="Glob pattern to discover episodes (default: *.mkv)")
    parser.add_argument("--tag-override", type=str, default=None,
                        help="Override episode tag for all frames (e.g. 'movie' for single files)")
    parser.add_argument("--val-dir", type=str, default=None,
                        help="Output directory for validation pairs")
    parser.add_argument("--num-frames", type=int, default=400,
                        help="Total training frames to extract (default: 400)")
    parser.add_argument("--num-val", type=int, default=40,
                        help="Total validation frames to extract (default: 40)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--sigma-min", type=float, default=2.0,
                        help="Minimum base sigma for blur (default: 2.0)")
    parser.add_argument("--sigma-max", type=float, default=5.0,
                        help="Maximum base sigma for blur (default: 5.0)")
    parser.add_argument("--downscale-prob", type=float, default=0.15,
                        help="Probability of downscale+upscale per frame (default: 0.15)")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: extract 10 pairs from first 2 episodes only")
    parser.add_argument("--no-skip", action="store_true",
                        help="Regenerate all frames (don't skip existing)")
    args = parser.parse_args()

    # Build episodes dict
    if args.files:
        episodes = episodes_from_files(args.files)
    else:
        episodes = discover_episodes(args.source_dir, args.pattern)

    if not episodes:
        print("ERROR: No video files found")
        sys.exit(1)

    sigma_range = (args.sigma_min, args.sigma_max)

    print(f"Name: {args.name}")
    print(f"Videos: {len(episodes)}")
    print(f"Sigma range: {args.sigma_min:.1f} - {args.sigma_max:.1f}")
    print(f"Downscale probability: {args.downscale_prob:.0%}")
    if args.tag_override:
        print(f"Tag override: {args.tag_override}")
    print()

    if args.test:
        test_eps = dict(list(sorted(episodes.items()))[:2])
        print(f"=== TEST MODE: 10 pairs from {list(test_eps.keys())} ===")
        print(f"Output: {args.output_dir}")
        n = generate_pairs(
            test_eps, args.name, args.output_dir,
            num_frames=10, seed=args.seed,
            sigma_range=sigma_range, downscale_prob=args.downscale_prob,
            skip_existing=not args.no_skip, tag_override=args.tag_override,
        )
        print(f"\nTest complete: {n} pairs generated in {args.output_dir}")
        return

    # Training set
    if args.num_frames > 0:
        print(f"=== TRAINING SET: {args.num_frames} frames ===")
        print(f"Output: {args.output_dir}")
        n_train = generate_pairs(
            episodes, args.name, args.output_dir,
            num_frames=args.num_frames, seed=args.seed,
            sigma_range=sigma_range, downscale_prob=args.downscale_prob,
            skip_existing=not args.no_skip, tag_override=args.tag_override,
        )
        print(f"\nTraining: {n_train} pairs in {args.output_dir}")
    else:
        n_train = 0

    # Validation set (different seed for no overlap)
    if args.val_dir and args.num_val > 0:
        print(f"\n=== VALIDATION SET: {args.num_val} frames ===")
        print(f"Output: {args.val_dir}")
        n_val = generate_pairs(
            episodes, args.name, args.val_dir,
            num_frames=args.num_val, seed=args.seed + 9999,
            sigma_range=sigma_range, downscale_prob=args.downscale_prob,
            skip_existing=not args.no_skip, tag_override=args.tag_override,
        )
        print(f"Validation: {n_val} pairs in {args.val_dir}")
    else:
        n_val = 0

    print(f"\nDone: {n_train} train + {n_val} val pairs")


if __name__ == "__main__":
    main()
