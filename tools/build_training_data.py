"""
Build training data in staged pipeline with proportional sampling.

Stages:
  --extract-only   Extract originals from video (1/500 frames), compute noise metrics
  --denoise        Denoise originals with adaptive sigma from calibration curve
  --build-inputs   Build degraded inputs (33% raw, 33% noise, 33% blur+noise)

All stages save/load from:
  data/originals/          Raw extracted frames + meta.pkl
  data/training/train/     Training pairs (input/ + target/)
  data/training/val/       Validation pairs (input/ + target/)
  data/calibration/        Sigma calibration artifacts

Usage:
    python tools/build_training_data.py --extract-only
    python tools/build_training_data.py --denoise
    python tools/build_training_data.py --build-inputs
    python tools/build_training_data.py --extract-only --test   # 2 frames per episode
"""
import sys
import os
import gc
import re
import random
import subprocess
import argparse
import time
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.ffmpeg_utils import get_ffmpeg, get_video_info

FFMPEG = get_ffmpeg()
DATA_DIR = Path("data")
ORIGINALS_DIR = DATA_DIR / "originals"
TRAINING_DIR = DATA_DIR / "training"
CALIBRATION_DIR = DATA_DIR / "calibration"

SAMPLE_RATE = 500  # 1 frame per N source frames
VAL_RATIO = 0.10
SEED = 42

# --- Source definitions ---
# Each source: source videos -> proportional frame extraction
# Pattern filtering excludes processed files (_nafnet, _gpu, _v2, _v3, etc.)
#
# Configure MEDIA_DIR environment variable to point to your video library.
# Example: export MEDIA_DIR=/path/to/media
# Sources below use relative paths under MEDIA_DIR.

MEDIA_DIR = Path(os.environ.get("MEDIA_DIR", "media"))

SOURCES = {
    "firefly": {
        "source_dir": str(MEDIA_DIR / "Firefly-S01"),
        "pattern": "*.mkv",
        "exclude_pattern": r"_(nafnet|gpu|v\d|raw)\.",
        "prefix": "firefly",
    },
    "expanse": {
        "source_dir": str(MEDIA_DIR / "Expanse-S02"),
        "pattern": "*.mkv",
        "prefix": "expanse",
    },
    "onepiece": {
        "source_dir": str(MEDIA_DIR / "OnePiece-S01"),
        "pattern": "*.mkv",
        "prefix": "onepiece",
    },
    "dune2": {
        "source_dir": str(MEDIA_DIR / "movies"),
        "pattern": "Dune*Part*Two*.mp4",
        "prefix": "dune2",
    },
    "squidgame": {
        "source_dir": str(MEDIA_DIR / "SquidGame-S02"),
        "pattern": "*.mkv",
        "prefix": "squidgame",
    },
    "foundation": {
        "source_dir": str(MEDIA_DIR / "Foundation-S03"),
        "pattern": "*.mkv",
        "prefix": "foundation",
    },
}

# NVDEC cuvid decoders
CUVID_DECODERS = {"h264": "h264_cuvid", "hevc": "hevc_cuvid"}


@dataclass
class VideoInfo:
    path: str
    tag: str
    width: int
    height: int
    fps: float
    total_frames: int
    duration: float
    codec: str
    cuvid_decoder: str | None


def probe_video(video_path: str, tag: str) -> VideoInfo:
    """Probe video for metadata using ffprobe/ffmpeg."""
    w, h, fps, total_frames, duration = get_video_info(video_path)

    # Detect codec for NVDEC
    r = subprocess.run(
        [FFMPEG, "-hide_banner", "-i", video_path],
        capture_output=True, text=True,
    )
    codec_match = re.search(r'Video:\s+(\w+)', r.stderr)
    codec = codec_match.group(1).lower() if codec_match else "unknown"

    return VideoInfo(
        path=video_path, tag=tag, width=w, height=h,
        fps=fps, total_frames=total_frames, duration=duration,
        codec=codec, cuvid_decoder=CUVID_DECODERS.get(codec),
    )


def discover_episodes(source_dir, pattern, exclude_pattern=None):
    """Discover video files, return {tag: full_path}. Excludes processed files."""
    import glob as glob_mod
    files = sorted(glob_mod.glob(os.path.join(glob_mod.escape(source_dir), pattern)))
    episodes = {}
    for f in files:
        basename = os.path.basename(f)
        # Skip processed/remastered files
        if exclude_pattern and re.search(exclude_pattern, basename):
            continue
        # Skip non-video directories
        if os.path.isdir(f):
            continue
        m = re.search(r'S(\d+)E(\d+)', basename, re.IGNORECASE)
        if m:
            tag = f"S{m.group(1).zfill(2)}E{m.group(2).zfill(2)}"
        else:
            tag = Path(basename).stem[:20].replace(" ", "_")
        episodes[tag] = os.path.abspath(f)
    return episodes


def episodes_from_files(file_paths):
    """Build episodes dict from explicit file paths."""
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


def extract_frame(info: VideoInfo, timestamp_sec: float) -> np.ndarray | None:
    """Extract a single frame at a timestamp using NVDEC hwaccel."""
    expected_bytes = info.width * info.height * 3

    def _run(use_hwaccel):
        cmd = [FFMPEG, "-hide_banner", "-loglevel", "error"]
        if use_hwaccel and info.cuvid_decoder:
            cmd += ["-hwaccel", "cuda", "-c:v", info.cuvid_decoder]
        cmd += [
            "-ss", f"{timestamp_sec:.3f}",
            "-i", info.path,
            "-frames:v", "1",
            "-f", "image2pipe", "-pix_fmt", "bgr24",
            "-vcodec", "rawvideo", "pipe:1",
        ]
        return subprocess.run(cmd, capture_output=True)

    r = _run(use_hwaccel=True)
    if (r.returncode != 0 or len(r.stdout) != expected_bytes) and info.cuvid_decoder:
        r = _run(use_hwaccel=False)

    if r.returncode != 0 or len(r.stdout) != expected_bytes:
        return None

    return np.frombuffer(r.stdout, dtype=np.uint8).reshape(info.height, info.width, 3)


def estimate_noise_level(frame_bgr: np.ndarray) -> float:
    """Estimate noise level from flat regions (low-gradient areas)."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sobel_sq = sobel_x ** 2 + sobel_y ** 2
    flat_mask = sobel_sq < np.percentile(sobel_sq, 25)
    if flat_mask.sum() < 1000:
        return 0.0
    return float(np.std(gray[flat_mask]))


def compute_frame_metrics(frame_bgr: np.ndarray) -> dict:
    """Compute noise_level, laplacian_var, sobel_mean for a frame."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # Laplacian variance
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = float(lap.var())

    gray_f = gray.astype(np.float32) / 255.0

    # Sobel gradient magnitude mean
    sobel_x = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    sobel_mean = float(sobel_mag.mean())

    # Noise level from flat regions
    sobel_sq = sobel_x ** 2 + sobel_y ** 2
    flat_mask = sobel_sq < np.percentile(sobel_sq, 25)
    if flat_mask.sum() < 1000:
        noise_level = 0.0
    else:
        noise_level = float(np.std(gray_f[flat_mask] * 255.0))

    return {
        "noise_level": noise_level,
        "laplacian_var": lap_var,
        "sobel_mean": sobel_mean,
    }


# ============================================================
# Stage 1: Extract originals
# ============================================================

def do_extract(args):
    """Extract original frames from all sources with proportional sampling."""
    os.makedirs(ORIGINALS_DIR, exist_ok=True)

    print("=== Stage 1: Extract Originals ===")
    print(f"Sample rate: 1 frame per {SAMPLE_RATE} source frames")
    print(f"Output: {ORIGINALS_DIR}")
    print()

    all_rows = []
    total_extracted = 0

    for source_name, config in sorted(SOURCES.items()):
        if args.only and args.only != source_name:
            continue

        prefix = config["prefix"]
        print(f"\n--- {source_name.upper()} ({prefix}) ---")

        # Discover episodes
        if "files" in config:
            episodes = episodes_from_files(config["files"])
        else:
            episodes = discover_episodes(
                config["source_dir"], config["pattern"],
                config.get("exclude_pattern"),
            )

        if not episodes:
            print(f"  ERROR: No video files found")
            continue

        # Probe all episodes in parallel
        video_infos = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                pool.submit(probe_video, path, tag): (tag, path)
                for tag, path in sorted(episodes.items())
            }
            for fut in as_completed(futures):
                tag, path = futures[fut]
                try:
                    info = fut.result()
                    video_infos.append(info)
                except Exception as e:
                    print(f"  SKIP {tag}: {e}")

        video_infos.sort(key=lambda v: v.tag)

        # Compute total frames across all episodes
        source_total_frames = sum(v.total_frames for v in video_infos)
        source_samples = max(1, source_total_frames // SAMPLE_RATE)
        if args.test:
            source_samples = min(source_samples, len(video_infos) * 2)

        print(f"  {len(video_infos)} episodes, {source_total_frames:,} total frames "
              f"-> {source_samples} samples")

        # Distribute samples proportionally across episodes
        episode_samples = []
        remaining = source_samples
        for i, info in enumerate(video_infos):
            if i == len(video_infos) - 1:
                n = remaining
            else:
                n = max(1, round(source_samples * info.total_frames / source_total_frames))
                n = min(n, remaining)
            episode_samples.append(n)
            remaining -= n

        # Extract frames from each episode
        rng = random.Random(SEED + hash(source_name))
        source_extracted = 0
        t0 = time.time()

        for info, n_frames in zip(video_infos, episode_samples):
            if n_frames <= 0:
                continue

            # Uniform timestamps across the full duration (skip first/last 60s for credits)
            start_t = min(60, info.duration * 0.05)
            end_t = max(info.duration - 60, info.duration * 0.95)
            if end_t <= start_t:
                start_t, end_t = 0, info.duration

            ep_rng = random.Random(rng.getrandbits(64))
            timestamps = sorted(ep_rng.uniform(start_t, end_t) for _ in range(n_frames))

            ep_count = 0
            for j, ts in enumerate(timestamps):
                filename = f"{prefix}_{info.tag}_{j:05d}.png"
                out_path = ORIGINALS_DIR / filename

                if out_path.exists() and not args.no_skip:
                    # Still need metrics if not in meta already
                    ep_count += 1
                    continue

                frame = extract_frame(info, ts)
                if frame is None:
                    continue

                cv2.imwrite(str(out_path), frame)
                ep_count += 1

            source_extracted += ep_count
            elapsed = time.time() - t0
            fps = source_extracted / elapsed if elapsed > 0 else 0
            hwaccel = f"NVDEC {info.cuvid_decoder}" if info.cuvid_decoder else "sw"
            print(f"    {info.tag}: {info.width}x{info.height} {hwaccel} - "
                  f"{ep_count}/{n_frames} frames ({fps:.1f} fps avg)")

        total_extracted += source_extracted
        print(f"  -> {source_extracted} frames extracted")

    print(f"\n{'='*50}")
    print(f"Total extracted: {total_extracted} frames in {ORIGINALS_DIR}")

    # Now compute metrics for all originals in parallel
    print(f"\nComputing per-frame metrics...")
    _compute_originals_meta()


def _compute_originals_meta():
    """Compute noise/sharpness metrics for all originals, save to meta.pkl."""
    meta_path = ORIGINALS_DIR / "meta.pkl"

    # Load existing meta if present (to skip already-measured frames)
    existing = {}
    if meta_path.exists():
        try:
            df_existing = pd.read_pickle(meta_path)
            existing = {row["filename"]: row for _, row in df_existing.iterrows()}
        except Exception:
            pass

    files = sorted(f for f in os.listdir(ORIGINALS_DIR) if f.endswith(".png"))
    to_measure = [f for f in files if f not in existing]

    print(f"  {len(files)} total frames, {len(to_measure)} need metrics")

    if not to_measure:
        print("  All frames already have metrics")
        return

    def _measure_one(filename):
        path = str(ORIGINALS_DIR / filename)
        img = cv2.imread(path)
        if img is None:
            return None
        metrics = compute_frame_metrics(img)
        h, w = img.shape[:2]
        metrics["filename"] = filename
        metrics["source"] = _classify_source(filename)
        metrics["width"] = w
        metrics["height"] = h
        return metrics

    rows = list(existing.values())
    t0 = time.time()
    done = 0

    # Parallel metric computation (CPU-bound but cv2 releases GIL)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_measure_one, f): f for f in to_measure}
        for fut in as_completed(futures):
            result = fut.result()
            if result is not None:
                rows.append(result)
            done += 1
            if done % 500 == 0:
                elapsed = time.time() - t0
                print(f"    {done}/{len(to_measure)} measured ({done/elapsed:.0f} fps)")

    elapsed = time.time() - t0
    print(f"  Measured {done} frames in {elapsed:.1f}s ({done/elapsed:.0f} fps)")

    df = pd.DataFrame(rows)
    df.to_pickle(meta_path)
    print(f"  Saved: {meta_path} ({len(df)} frames)")

    # Summary
    print(f"\n  {'Source':<20} {'Count':>6} {'Noise Mean':>12} {'Noise Med':>12}")
    print(f"  {'-'*20} {'-'*6} {'-'*12} {'-'*12}")
    for src in sorted(df["source"].unique()):
        sub = df[df["source"] == src]
        print(f"  {src:<20} {len(sub):>6} "
              f"{sub['noise_level'].mean():>12.2f} {sub['noise_level'].median():>12.2f}")


def _classify_source(filename):
    """Extract source name from filename prefix."""
    # Filenames: prefix_tag_00000.png
    parts = filename.split("_")
    return parts[0]


# ============================================================
# Stage 2: Denoise with SCUNet GAN + unsharp mask
# ============================================================

USM_STRENGTH = 1.0  # unsharp mask strength applied after SCUNet GAN
USM_SIGMA = 1.5     # unsharp mask blur sigma


def load_scunet_gan(device="cuda"):
    """Load SCUNet GAN model (denoise + sharpen in one pass)."""
    from lib.paths import resolve_scunet_dir
    scunet_dir = resolve_scunet_dir()
    sys.path.insert(0, str(scunet_dir))
    from models.network_scunet import SCUNet

    weights_path = os.path.join(str(scunet_dir), "model_zoo", "scunet_color_real_gan.pth")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"SCUNet GAN weights not found at {weights_path}")

    model = SCUNet(in_nc=3, config=[4, 4, 4, 4, 4, 4, 4], dim=64)
    ckpt = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(ckpt, strict=True)
    del ckpt; gc.collect()

    model = model.half().to(device).eval()
    print(f"SCUNet GAN loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params, "
          f"VRAM: {torch.cuda.memory_allocated()/1024**2:.0f}MB")
    return model


@torch.no_grad()
def scunet_denoise(model, frame_bgr, device="cuda"):
    """Denoise+sharpen a BGR frame with SCUNet GAN."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    tensor = tensor.half().to(device)

    _, _, h, w = tensor.shape
    # SCUNet needs dimensions divisible by 64 (Swin Transformer windows)
    pad_h = (64 - h % 64) % 64
    pad_w = (64 - w % 64) % 64
    if pad_h or pad_w:
        tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')

    output = model(tensor)

    if pad_h or pad_w:
        output = output[:, :, :h, :w]

    output = output.squeeze(0).clamp(0, 1).float().cpu().numpy()
    output = (output * 255.0).round().astype(np.uint8)
    output = np.transpose(output, (1, 2, 0))
    return cv2.cvtColor(output, cv2.COLOR_RGB2BGR)


def unsharp_mask(img_bgr, strength=USM_STRENGTH, sigma=USM_SIGMA):
    """Light unsharp mask to push detail slightly beyond SCUNet GAN output."""
    if strength <= 0:
        return img_bgr
    f = img_bgr.astype(np.float32)
    blurred = cv2.GaussianBlur(f, (0, 0), sigmaX=sigma)
    result = f + strength * (f - blurred)
    return np.clip(result, 0, 255).astype(np.uint8)


def do_denoise(args):
    """Denoise originals with SCUNet GAN + light USM -> targets."""
    meta_path = ORIGINALS_DIR / "meta.pkl"
    if not meta_path.exists():
        print("ERROR: Run --extract-only first to create originals and meta.pkl")
        sys.exit(1)

    df = pd.read_pickle(meta_path)
    print(f"=== Stage 2: Denoise with SCUNet GAN + USM({USM_STRENGTH}) ===")
    print(f"Loaded {len(df)} frames from {meta_path}")

    # Train/val split (fixed seed, stratified by source)
    rng = random.Random(SEED)
    train_files = []
    val_files = []
    for source in sorted(df["source"].unique()):
        source_df = df[df["source"] == source]
        indices = list(source_df.index)
        rng.shuffle(indices)
        n_val = max(1, int(len(indices) * VAL_RATIO))
        val_files.extend(indices[:n_val])
        train_files.extend(indices[n_val:])

    df["split"] = "train"
    df.loc[val_files, "split"] = "val"

    # Verify no overlap
    train_set = set(df.loc[df["split"] == "train", "filename"])
    val_set = set(df.loc[df["split"] == "val", "filename"])
    assert len(train_set & val_set) == 0, "Train/val overlap detected!"
    assert len(train_set) + len(val_set) == len(df), "Missing frames in split!"
    print(f"Split: {len(train_set)} train + {len(val_set)} val (verified no overlap)")

    # Per-source breakdown
    for source in sorted(df["source"].unique()):
        sub = df[df["source"] == source]
        n_train = (sub["split"] == "train").sum()
        n_val = (sub["split"] == "val").sum()
        print(f"  {source:<15} {n_train:>5} train + {n_val:>4} val")

    # Create output dirs
    for split in ("train", "val"):
        os.makedirs(TRAINING_DIR / split / "target", exist_ok=True)
        os.makedirs(TRAINING_DIR / split / "input", exist_ok=True)

    # Load SCUNet GAN
    model = load_scunet_gan()

    t0 = time.time()
    done = 0
    skipped = 0

    for idx, row in df.iterrows():
        filename = row["filename"]
        split = row["split"]
        target_path = TRAINING_DIR / split / "target" / filename

        if target_path.exists() and not args.no_skip:
            skipped += 1
            continue

        orig_path = ORIGINALS_DIR / filename
        frame = cv2.imread(str(orig_path))
        if frame is None:
            print(f"  WARN: could not read {filename}")
            continue

        # SCUNet GAN: denoise + sharpen in one pass
        denoised = scunet_denoise(model, frame)
        # Light USM to push detail a touch further
        target = unsharp_mask(denoised)

        cv2.imwrite(str(target_path), target)
        done += 1

        if done % 100 == 0:
            elapsed = time.time() - t0
            fps = done / elapsed if elapsed > 0 else 0
            print(f"  {done}/{len(df) - skipped} processed ({fps:.1f} fps)")

    elapsed = time.time() - t0
    if done > 0:
        print(f"\nProcessed {done} frames in {elapsed:.1f}s ({done/elapsed:.1f} fps)")
    if skipped:
        print(f"Skipped {skipped} existing targets")

    # Save training meta
    training_meta_path = TRAINING_DIR / "meta.pkl"
    df.to_pickle(training_meta_path)
    print(f"Saved: {training_meta_path}")


# ============================================================
# Stage 3: Build inputs
# ============================================================

def edge_aware_degrade(original, clean_target, base_sigma, rng, downscale_prob=0.15):
    """Apply edge-aware degradation to the original frame.

    Edge map from clean target (noise-free edge detection).
    Blur applied to original (retains real artifacts).
    """
    h, w = original.shape[:2]
    img_f = original.astype(np.float32) / 255.0

    # Compute edge map on CLEAN target
    gray_clean = cv2.cvtColor(clean_target, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    sobel_x = cv2.Sobel(gray_clean, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_clean, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    edge_mag = edge_mag / (edge_mag.max() + 1e-8)

    # Smooth edge map for gradual transitions
    edge_map = cv2.GaussianBlur(edge_mag, (0, 0), sigmaX=5.0)
    edge_map = edge_map / (edge_map.max() + 1e-8)

    # Spatially-varying blur applied to ORIGINAL frame
    max_sigma = base_sigma * 2.0
    ksize = int(max_sigma * 6) | 1
    ksize = max(ksize, 3)
    blurred = cv2.GaussianBlur(img_f, (ksize, ksize), sigmaX=max_sigma)

    alpha = edge_map[:, :, np.newaxis]
    alpha = alpha * min(base_sigma / 2.0, 1.0)
    degraded = img_f * (1.0 - alpha) + blurred * alpha

    # Optional downscale + upscale
    if rng.random() < downscale_prob:
        scale = rng.uniform(0.5, 0.75)
        small_h, small_w = int(h * scale), int(w * scale)
        degraded = cv2.resize(degraded, (small_w, small_h), interpolation=cv2.INTER_AREA)
        degraded = cv2.resize(degraded, (w, h), interpolation=cv2.INTER_CUBIC)

    # Add light Gaussian noise
    noise_sigma = rng.uniform(1.0, 5.0) / 255.0
    np_rng = np.random.default_rng(rng.getrandbits(64))
    noise = np_rng.standard_normal(degraded.shape, dtype=np.float32) * noise_sigma
    degraded = degraded + noise

    degraded = np.clip(degraded * 255.0, 0, 255).astype(np.uint8)
    return degraded


def _build_input_for_frame(filename, split, rng_seed):
    """Build a single degraded input frame. Returns degradation_type or None on error."""
    orig_path = ORIGINALS_DIR / filename
    target_path = TRAINING_DIR / split / "target" / filename
    input_path = TRAINING_DIR / split / "input" / filename

    original = cv2.imread(str(orig_path))
    if original is None:
        return None

    rng = random.Random(rng_seed)
    choice = rng.random()

    if choice < 1.0 / 3.0:
        # Raw original unchanged
        cv2.imwrite(str(input_path), original)
        return "raw"

    elif choice < 2.0 / 3.0:
        # Original + light Gaussian noise
        img_f = original.astype(np.float32) / 255.0
        noise_sigma = rng.uniform(1.0, 5.0) / 255.0
        np_rng = np.random.default_rng(rng.getrandbits(64))
        noise = np_rng.standard_normal(img_f.shape, dtype=np.float32) * noise_sigma
        noisy = np.clip((img_f + noise) * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(str(input_path), noisy)
        return "noise"

    else:
        # Original + edge-aware blur + noise
        target = cv2.imread(str(target_path))
        if target is None:
            # Fall back to raw if target missing
            cv2.imwrite(str(input_path), original)
            return "raw"
        base_sigma = rng.uniform(2.0, 8.0)
        degraded = edge_aware_degrade(original, target, base_sigma, rng)
        cv2.imwrite(str(input_path), degraded)
        return "blur_noise"


def do_build_inputs(args):
    """Build degraded input frames from originals (parallel CPU)."""
    training_meta_path = TRAINING_DIR / "meta.pkl"
    if not training_meta_path.exists():
        print("ERROR: Run --denoise first to create targets and meta.pkl")
        sys.exit(1)

    df = pd.read_pickle(training_meta_path)
    print(f"=== Stage 3: Build Inputs ===")
    print(f"Loaded {len(df)} frames from {training_meta_path}")

    # Create output dirs
    for split in ("train", "val"):
        os.makedirs(TRAINING_DIR / split / "input", exist_ok=True)

    # Determine which frames need inputs
    work_items = []
    rng = random.Random(SEED + 12345)
    for idx, row in df.iterrows():
        filename = row["filename"]
        split = row["split"]
        input_path = TRAINING_DIR / split / "input" / filename

        if input_path.exists() and not args.no_skip:
            continue

        work_items.append((filename, split, rng.getrandbits(64)))

    print(f"  {len(work_items)} inputs to build ({len(df) - len(work_items)} already exist)")

    if not work_items:
        print("  All inputs already exist")
        return

    t0 = time.time()
    done = 0
    degradation_types = []

    # Parallel input building (CPU-bound: blur, noise, I/O)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {}
        for filename, split, seed in work_items:
            fut = pool.submit(_build_input_for_frame, filename, split, seed)
            futures[fut] = filename

        for fut in as_completed(futures):
            result = fut.result()
            if result is not None:
                degradation_types.append((futures[fut], result))
            done += 1
            if done % 500 == 0:
                elapsed = time.time() - t0
                print(f"    {done}/{len(work_items)} ({done/elapsed:.0f} fps)")

    elapsed = time.time() - t0
    print(f"\nBuilt {done} inputs in {elapsed:.1f}s ({done/elapsed:.0f} fps)")

    # Update meta with degradation types
    deg_map = dict(degradation_types)
    if "degradation_type" not in df.columns:
        df["degradation_type"] = ""
    for filename, deg_type in deg_map.items():
        mask = df["filename"] == filename
        df.loc[mask, "degradation_type"] = deg_type

    df.to_pickle(training_meta_path)
    print(f"Updated: {training_meta_path}")

    # Summary
    if degradation_types:
        from collections import Counter
        counts = Counter(t for _, t in degradation_types)
        print(f"\nDegradation types:")
        for t, c in sorted(counts.items()):
            print(f"  {t}: {c} ({100*c/len(degradation_types):.0f}%)")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build training data in staged pipeline")
    parser.add_argument("--extract-only", action="store_true",
                        help="Stage 1: Extract originals + compute metrics")
    parser.add_argument("--denoise", action="store_true",
                        help="Stage 2: Denoise originals with adaptive sigma")
    parser.add_argument("--build-inputs", action="store_true",
                        help="Stage 3: Build degraded input frames")
    parser.add_argument("--only", type=str, default=None,
                        help="Process only this source (e.g. 'firefly', 'expanse')")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: minimal frames per episode")
    parser.add_argument("--no-skip", action="store_true",
                        help="Regenerate all frames (don't skip existing)")
    args = parser.parse_args()

    if not any([args.extract_only, args.denoise, args.build_inputs]):
        parser.print_help()
        print("\nERROR: Specify a stage: --extract-only, --denoise, or --build-inputs")
        sys.exit(1)

    if args.extract_only:
        do_extract(args)
    elif args.denoise:
        do_denoise(args)
    elif args.build_inputs:
        do_build_inputs(args)


if __name__ == "__main__":
    main()
