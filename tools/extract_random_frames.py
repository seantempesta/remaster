"""
Extract randomly sampled frames from multiple video files.

Randomly seeks to positions throughout each video and extracts single frames.
This gives better diversity than sequential extraction.

Usage:
    python tools/extract_random_frames.py --episodes E02 E03 E05 --frames-per-ep 30 --output-dir data/frames_new
    python tools/extract_random_frames.py --all --frames-per-ep 30 --output-dir data/frames_new
"""
import sys
import os
import random
import subprocess
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.ffmpeg_utils import get_ffmpeg, get_ffprobe

FFMPEG = get_ffmpeg()
PLEX_DIR = r"E:\plex\tv\Firefly (2002) Season 1 S01 (1080p BluRay x265 HEVC 10bit AAC Silence)"

# Map episode numbers to filenames
EPISODES = {
    "E01": "Firefly (2002) - S01E01 - Serenity (1080p x265 Silence).mkv",
    "E02": "Firefly (2002) - S01E02 - The Train Job (1080p BluRay x265 Silence).mkv",
    "E03": "Firefly (2002) - S01E03 - Bushwhacked (1080p BluRay x265 Silence).mkv",
    "E04": "Firefly (2002) - S01E04 - Shindig (1080p BluRay x265 Silence).mkv",
    "E05": "Firefly (2002) - S01E05 - Safe (1080p BluRay x265 Silence).mkv",
    "E06": "Firefly (2002) - S01E06 - Our Mrs. Reynolds (1080p BluRay x265 Silence).mkv",
    "E07": "Firefly (2002) - S01E07 - Jaynestown (1080p BluRay x265 Silence).mkv",
    "E08": "Firefly (2002) - S01E08 - Out of Gas (1080p BluRay x265 Silence).mkv",
    "E09": "Firefly (2002) - S01E09 - Ariel (1080p BluRay x265 Silence).mkv",
    "E10": "Firefly (2002) - S01E10 - War Stories (1080p BluRay x265 Silence).mkv",
    "E11": "Firefly (2002) - S01E11 - Trash (1080p BluRay x265 Silence).mkv",
    "E12": "Firefly (2002) - S01E12 - The Message (1080p BluRay x265 Silence).mkv",
    "E13": "Firefly (2002) - S01E13 - Heart of Gold (1080p BluRay x265 Silence).mkv",
    "E14": "Firefly (2002) - S01E14 - Objects in Space (1080p BluRay x265 Silence).mkv",
}


def get_duration(path):
    """Get video duration in seconds using ffprobe."""
    ffprobe = get_ffprobe()
    if ffprobe:
        r = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", path],
            capture_output=True, text=True,
        )
        try:
            return float(r.stdout.strip())
        except ValueError:
            pass
    # Fallback: use ffmpeg to probe
    r = subprocess.run(
        [FFMPEG, "-hide_banner", "-i", path],
        capture_output=True, text=True,
    )
    for line in r.stderr.split("\n"):
        if "Duration:" in line:
            parts = line.split("Duration:")[1].split(",")[0].strip()
            h, m, s = parts.split(":")
            return float(h) * 3600 + float(m) * 60 + float(s)
    return 2400  # fallback: 40 min


def extract_random_frames(video_path, output_dir, episode_tag, num_frames=30, seed=42):
    """Extract num_frames randomly sampled frames from a video."""
    duration = get_duration(video_path)
    # Skip first/last 60 seconds (credits/black)
    start = 60
    end = duration - 60
    if end <= start:
        start, end = 0, duration

    rng = random.Random(seed)
    timestamps = sorted(rng.uniform(start, end) for _ in range(num_frames))

    os.makedirs(output_dir, exist_ok=True)
    extracted = 0

    for i, ts in enumerate(timestamps):
        outfile = os.path.join(output_dir, f"{episode_tag}_{i + 1:05d}.png")
        if os.path.exists(outfile):
            extracted += 1
            continue

        # Seek to timestamp, extract single frame
        r = subprocess.run(
            [FFMPEG, "-hide_banner", "-y",
             "-ss", f"{ts:.3f}", "-i", video_path,
             "-frames:v", "1", "-q:v", "1",
             outfile],
            capture_output=True, text=True,
        )
        if r.returncode == 0 and os.path.exists(outfile):
            extracted += 1
        else:
            print(f"  WARN: failed to extract frame at {ts:.1f}s: {r.stderr[-200:]}")

    return extracted


def main():
    parser = argparse.ArgumentParser(description="Extract random frames from Firefly episodes")
    parser.add_argument("--episodes", nargs="+", default=None,
                        help="Episode tags (E02, E03, ...). Default: all untrained episodes")
    parser.add_argument("--all", action="store_true",
                        help="Use all 14 episodes")
    parser.add_argument("--frames-per-ep", type=int, default=30,
                        help="Frames to extract per episode (default: 30)")
    parser.add_argument("--output-dir", type=str, default="data/frames_new",
                        help="Output directory for extracted frames")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.all:
        episodes = list(EPISODES.keys())
    elif args.episodes:
        episodes = [e.upper() for e in args.episodes]
    else:
        # Default: episodes not already in training data
        episodes = ["E02", "E03", "E05", "E06", "E07", "E09", "E10", "E11", "E12", "E13", "E14"]

    print(f"Extracting {args.frames_per_ep} random frames from {len(episodes)} episodes")
    print(f"Output: {args.output_dir}")
    print()

    total = 0
    for ep in episodes:
        if ep not in EPISODES:
            print(f"  SKIP: unknown episode {ep}")
            continue
        video_path = os.path.join(PLEX_DIR, EPISODES[ep])
        if not os.path.exists(video_path):
            print(f"  SKIP: {ep} not found at {video_path}")
            continue

        print(f"  {ep}: {EPISODES[ep]}")
        n = extract_random_frames(
            video_path, args.output_dir, ep.lower(),
            num_frames=args.frames_per_ep,
            seed=args.seed + hash(ep),
        )
        print(f"    -> {n} frames extracted")
        total += n

    print(f"\nDone: {total} frames in {args.output_dir}")


if __name__ == "__main__":
    main()
