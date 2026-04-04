"""
Extract synthetic training pairs from high-quality Bluray source material.

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
    python tools/extract_synthetic_pairs.py --test          # 10 test pairs
    python tools/extract_synthetic_pairs.py                 # full 1200 + 30 val
    python tools/extract_synthetic_pairs.py --source-dir "E:/plex/tv/Show" --episode-pattern "*.mkv"
"""
import sys
import os
import random
import subprocess
import argparse
import glob
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.ffmpeg_utils import get_ffmpeg, get_ffprobe

FFMPEG = get_ffmpeg()

# Default source: The Expanse Season 2
DEFAULT_SOURCE_DIR = r"E:\plex\tv\The Expanse Season 2  [1080p Bluray x265 q22 S96 Joy]"
DEFAULT_OUTPUT_DIR = "data/synthetic_pairs"
DEFAULT_VAL_DIR = "data/synthetic_val"

# Episodes map (auto-discovered if using --episode-pattern)
EXPANSE_EPISODES = {
    "S02E01": "The Expanse S02E01 Safe  (1080p x265 Q22 S96 Joy).mkv",
    "S02E02": "The Expanse S02E02 Doors & Corners  (1080p x265 q22 Joy).mkv",
    "S02E03": "The Expanse S02E03 Static  (1080p x265 q22 Joy).mkv",
    "S02E04": "The Expanse S02E04 Godspeed  (1080p x265 q22 Joy).mkv",
    "S02E05": "The Expanse S02E05 Home  (1080p x265 q22 Joy).mkv",
    "S02E06": "The Expanse S02E06 Paradigm Shift  (1080p x265 q22 Joy).mkv",
    "S02E07": "The Expanse S02E07 The Seventh Man  (1080p x265 q22 Joy).mkv",
    "S02E08": "The Expanse S02E08 Pyre  (1080p x265 q22 Joy).mkv",
    "S02E09": "The Expanse S02E09 The Weeping Somnambulist  (1080p x265 q22 Joy).mkv",
    "S02E10": "The Expanse S02E10 Cascade  (1080p x265 q22 Joy).mkv",
    "S02E11": "The Expanse S02E11 Here There Be Dragons  (1080p x265 q22 Joy).mkv",
    "S02E12": "The Expanse S02E12 The Monster and the Rocket  (1080p x265 q22 Joy).mkv",
    "S02E13": "The Expanse S02E13 Caliban's War  (1080p x265 q22 Joy).mkv",
}


def get_duration(video_path):
    """Get video duration in seconds."""
    ffprobe = get_ffprobe()
    if ffprobe:
        r = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", video_path],
            capture_output=True, text=True,
        )
        try:
            return float(r.stdout.strip())
        except ValueError:
            pass
    # Fallback
    r = subprocess.run(
        [FFMPEG, "-hide_banner", "-i", video_path],
        capture_output=True, text=True,
    )
    for line in r.stderr.split("\n"):
        if "Duration:" in line:
            parts = line.split("Duration:")[1].split(",")[0].strip()
            h, m, s = parts.split(":")
            return float(h) * 3600 + float(m) * 60 + float(s)
    return 2400


def extract_frame_to_array(video_path, timestamp_sec):
    """Extract a single frame at a timestamp, return as numpy array (BGR).

    Uses ffmpeg to decode a single frame and pipe raw pixels to stdout.
    No temp files needed.
    """
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
    if r.returncode != 0 or len(r.stdout) == 0:
        return None

    # We need dimensions to reshape — get from video info
    # For 1080p content this is always 1920x1080
    # Parse actual dimensions from the raw data size
    raw_bytes = r.stdout
    n_pixels = len(raw_bytes) // 3
    # Try common resolutions
    for w, h in [(1920, 1080), (1280, 720), (3840, 2160), (1920, 800), (1920, 816)]:
        if w * h == n_pixels:
            return np.frombuffer(raw_bytes, dtype=np.uint8).reshape(h, w, 3)

    # Fallback: assume 1920 width
    if n_pixels >= 1920:
        h = n_pixels // 1920
        if h * 1920 == n_pixels:
            return np.frombuffer(raw_bytes, dtype=np.uint8).reshape(h, 1920, 3)

    return None


def edge_aware_degrade(image, base_sigma, rng, downscale_prob=0.15):
    """Apply edge-aware degradation to an image.

    Sharp areas (high edge magnitude) get blurred more.
    Soft areas (low edge magnitude) stay mostly untouched.

    Args:
        image: BGR uint8 numpy array
        base_sigma: base blur strength (typically 1.0-2.5)
        rng: random.Random instance
        downscale_prob: probability of applying downscale+upscale

    Returns:
        degraded BGR uint8 numpy array
    """
    h, w = image.shape[:2]

    # Convert to float32 for processing
    img_f = image.astype(np.float32) / 255.0

    # --- Compute edge map ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    # Normalize edge map to [0, 1]
    edge_mag = edge_mag / (edge_mag.max() + 1e-8)

    # Smooth the edge map so blur transitions are gradual (not pixel-sharp)
    edge_map = cv2.GaussianBlur(edge_mag, (0, 0), sigmaX=5.0)
    edge_map = edge_map / (edge_map.max() + 1e-8)

    # --- Apply spatially-varying blur ---
    # Strategy: blend between original and strongly-blurred version,
    # weighted by edge map. This is more efficient than per-pixel sigma.
    # Strong blur for the "max degradation" version
    max_sigma = base_sigma * 2.0
    ksize = int(max_sigma * 6) | 1  # ensure odd
    ksize = max(ksize, 3)
    blurred = cv2.GaussianBlur(img_f, (ksize, ksize), sigmaX=max_sigma)

    # Edge map controls blend: high edge = more blur, low edge = keep original
    # Expand edge_map to 3 channels
    alpha = edge_map[:, :, np.newaxis]

    # Scale alpha by base_sigma ratio so lower base_sigma = subtler effect
    alpha = alpha * min(base_sigma / 2.0, 1.0)

    degraded = img_f * (1.0 - alpha) + blurred * alpha

    # --- Optional downscale + upscale (simulates resolution loss) ---
    if rng.random() < downscale_prob:
        scale = rng.uniform(0.5, 0.75)
        small_h, small_w = int(h * scale), int(w * scale)
        degraded = cv2.resize(degraded, (small_w, small_h), interpolation=cv2.INTER_AREA)
        degraded = cv2.resize(degraded, (w, h), interpolation=cv2.INTER_CUBIC)

    # Clip and convert back to uint8
    degraded = np.clip(degraded * 255.0, 0, 255).astype(np.uint8)
    return degraded


def discover_episodes(source_dir, pattern="*.mkv"):
    """Auto-discover episode files matching a glob pattern."""
    import re
    files = sorted(glob.glob(os.path.join(source_dir, pattern)))
    episodes = {}
    for f in files:
        basename = os.path.basename(f)
        # Try to extract SxxExx pattern
        m = re.search(r'S(\d+)E(\d+)', basename, re.IGNORECASE)
        if m:
            tag = f"S{m.group(1).zfill(2)}E{m.group(2).zfill(2)}"
        else:
            # Use filename stem as tag
            tag = Path(basename).stem[:20].replace(" ", "_")
        episodes[tag] = basename
    return episodes


def generate_pairs(source_dir, episodes, output_dir, num_frames_total,
                   seed=42, sigma_range=(1.0, 2.5), downscale_prob=0.15,
                   skip_existing=True):
    """Extract frames and generate degraded pairs.

    Args:
        source_dir: directory containing video files
        episodes: dict of {tag: filename}
        output_dir: base output dir (will create input/ and target/ subdirs)
        num_frames_total: total frames to extract across all episodes
        seed: random seed
        sigma_range: (min, max) base_sigma for degradation
        downscale_prob: probability of downscale+upscale per frame
        skip_existing: skip frames that already exist on disk

    Returns:
        number of frames successfully generated
    """
    input_dir = os.path.join(output_dir, "input")
    target_dir = os.path.join(output_dir, "target")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)

    rng = random.Random(seed)
    num_episodes = len(episodes)
    frames_per_ep = num_frames_total // num_episodes
    remainder = num_frames_total % num_episodes

    total_generated = 0
    total_skipped = 0

    for i, (tag, filename) in enumerate(sorted(episodes.items())):
        video_path = os.path.join(source_dir, filename)
        if not os.path.exists(video_path):
            print(f"  SKIP: {tag} not found at {video_path}")
            continue

        # Distribute remainder across first N episodes
        n_frames = frames_per_ep + (1 if i < remainder else 0)

        duration = get_duration(video_path)
        # Skip first/last 90 seconds (credits, black frames)
        start_t = 90
        end_t = duration - 90
        if end_t <= start_t:
            start_t, end_t = 30, duration - 30
        if end_t <= start_t:
            start_t, end_t = 0, duration

        # Generate random timestamps
        ep_rng = random.Random(seed + hash(tag))
        timestamps = sorted(ep_rng.uniform(start_t, end_t) for _ in range(n_frames))

        ep_generated = 0
        ep_skipped = 0

        for j, ts in enumerate(timestamps):
            frame_tag = f"expanse_{tag}_{j:05d}"
            input_path = os.path.join(input_dir, f"{frame_tag}.png")
            target_path = os.path.join(target_dir, f"{frame_tag}.png")

            if skip_existing and os.path.exists(input_path) and os.path.exists(target_path):
                ep_skipped += 1
                continue

            # Extract frame
            frame = extract_frame_to_array(video_path, ts)
            if frame is None:
                print(f"    WARN: failed to extract {tag} at {ts:.1f}s")
                continue

            # Random sigma for this frame
            base_sigma = rng.uniform(*sigma_range)

            # Generate degraded version
            degraded = edge_aware_degrade(frame, base_sigma, rng,
                                          downscale_prob=downscale_prob)

            # Save pair
            cv2.imwrite(target_path, frame)
            cv2.imwrite(input_path, degraded)
            ep_generated += 1

        total_generated += ep_generated
        total_skipped += ep_skipped
        status = f"{ep_generated} new"
        if ep_skipped:
            status += f", {ep_skipped} skipped"
        print(f"  {tag}: {status} (sigma {sigma_range[0]:.1f}-{sigma_range[1]:.1f})")

    return total_generated


def main():
    parser = argparse.ArgumentParser(
        description="Extract synthetic training pairs with edge-aware degradation")
    parser.add_argument("--source-dir", type=str, default=DEFAULT_SOURCE_DIR,
                        help="Directory containing source video files")
    parser.add_argument("--episode-pattern", type=str, default=None,
                        help="Glob pattern to discover episodes (e.g. '*.mkv')")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Output directory for training pairs")
    parser.add_argument("--val-dir", type=str, default=DEFAULT_VAL_DIR,
                        help="Output directory for validation pairs")
    parser.add_argument("--num-frames", type=int, default=1200,
                        help="Total training frames to extract (default: 1200)")
    parser.add_argument("--num-val", type=int, default=30,
                        help="Total validation frames to extract (default: 30)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--sigma-min", type=float, default=1.0,
                        help="Minimum base sigma for blur (default: 1.0)")
    parser.add_argument("--sigma-max", type=float, default=2.5,
                        help="Maximum base sigma for blur (default: 2.5)")
    parser.add_argument("--downscale-prob", type=float, default=0.15,
                        help="Probability of downscale+upscale per frame (default: 0.15)")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: extract 10 pairs from first 2 episodes only")
    parser.add_argument("--no-skip", action="store_true",
                        help="Regenerate all frames (don't skip existing)")
    args = parser.parse_args()

    # Discover or use hardcoded episodes
    if args.episode_pattern:
        episodes = discover_episodes(args.source_dir, args.episode_pattern)
    else:
        episodes = EXPANSE_EPISODES

    if not episodes:
        print("ERROR: No episodes found")
        sys.exit(1)

    print(f"Source: {args.source_dir}")
    print(f"Episodes: {len(episodes)}")
    print(f"Sigma range: {args.sigma_min:.1f} - {args.sigma_max:.1f}")
    print(f"Downscale probability: {args.downscale_prob:.0%}")
    print()

    sigma_range = (args.sigma_min, args.sigma_max)

    if args.test:
        # Test mode: 10 pairs from first 2 episodes
        test_eps = dict(list(sorted(episodes.items()))[:2])
        print(f"=== TEST MODE: 10 pairs from {list(test_eps.keys())} ===")
        print(f"Output: {args.output_dir}")
        n = generate_pairs(
            args.source_dir, test_eps, args.output_dir,
            num_frames_total=10, seed=args.seed,
            sigma_range=sigma_range, downscale_prob=args.downscale_prob,
            skip_existing=not args.no_skip,
        )
        print(f"\nTest complete: {n} pairs generated in {args.output_dir}")
        return

    # Full extraction: training set
    print(f"=== TRAINING SET: {args.num_frames} frames ===")
    print(f"Output: {args.output_dir}")
    n_train = generate_pairs(
        args.source_dir, episodes, args.output_dir,
        num_frames_total=args.num_frames, seed=args.seed,
        sigma_range=sigma_range, downscale_prob=args.downscale_prob,
        skip_existing=not args.no_skip,
    )
    print(f"\nTraining: {n_train} pairs in {args.output_dir}")

    # Validation set (different seed so no overlap)
    print(f"\n=== VALIDATION SET: {args.num_val} frames ===")
    print(f"Output: {args.val_dir}")
    n_val = generate_pairs(
        args.source_dir, episodes, args.val_dir,
        num_frames_total=args.num_val, seed=args.seed + 9999,
        sigma_range=sigma_range, downscale_prob=args.downscale_prob,
        skip_existing=not args.no_skip,
    )
    print(f"Validation: {n_val} pairs in {args.val_dir}")

    print(f"\nDone: {n_train} train + {n_val} val pairs")


if __name__ == "__main__":
    main()
