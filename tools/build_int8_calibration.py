"""
Build INT8 calibration data for TensorRT.

Extracts random full-resolution frames directly from source videos
(the actual content the model will process in production).

The calibration cache is portable - works with trtexec, C++ TRT API, or vs-mlrt.

Usage:
  python tools/build_int8_calibration.py
  python tools/build_int8_calibration.py --num-frames 200
  python tools/build_int8_calibration.py --data-only  # skip engine build
"""
import argparse
import os
import sys
import random
import subprocess
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def find_source_videos():
    """Find original source videos (the actual content we remaster)."""
    videos = []

    # Plex library - the real source material
    plex_dir = Path("E:/plex/tv")
    if plex_dir.exists():
        for mkv in plex_dir.rglob("*.mkv"):
            # Skip any remastered/processed versions
            name = mkv.stem.lower()
            if any(skip in name for skip in ["_gpu", "_nafnet", "_student", "_teacher", "remaster", ".raw"]):
                continue
            videos.append(str(mkv))

    # Archive clips (30s test clips from various sources)
    archive_dir = PROJECT_ROOT / "data" / "archive"
    if archive_dir.exists():
        for mkv in archive_dir.glob("*_30s.mkv"):
            videos.append(str(mkv))

    print(f"Found {len(videos)} source videos")
    for v in videos[:10]:
        print(f"  {Path(v).name}")
    if len(videos) > 10:
        print(f"  ... and {len(videos) - 10} more")

    return videos


def extract_random_frames(videos, num_frames=200, target_h=1080, target_w=1920):
    """Extract random frames from source videos using PyAV (CPU decode)."""
    import av

    # Distribute frames across videos proportionally to duration
    video_durations = []
    for v in videos:
        try:
            container = av.open(v)
            dur = float(container.duration) / av.time_base if container.duration else 0
            stream = container.streams.video[0]
            dur = stream.frames / stream.average_rate if stream.frames and stream.average_rate else dur
            video_durations.append((v, max(dur, 1)))
            container.close()
        except Exception as e:
            print(f"  Skip {Path(v).name}: {e}")

    total_dur = sum(d for _, d in video_durations)
    frames_per_video = {v: max(1, int(num_frames * d / total_dur))
                        for v, d in video_durations}

    # Adjust to hit target count
    assigned = sum(frames_per_video.values())
    if assigned < num_frames:
        # Add extras to longest videos
        for v, _ in sorted(video_durations, key=lambda x: -x[1]):
            if assigned >= num_frames:
                break
            frames_per_video[v] += 1
            assigned += 1

    print(f"\nExtracting {num_frames} frames from {len(frames_per_video)} videos...")

    random.seed(42)
    all_frames = []

    for video_path, n_frames in frames_per_video.items():
        try:
            container = av.open(video_path)
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            total = stream.frames or 10000

            # Pick random frame indices
            indices = sorted(random.sample(range(max(10, total - 10)), min(n_frames, total - 20)))

            frame_count = 0
            extracted = 0
            for frame in container.decode(stream):
                if extracted >= len(indices):
                    break
                if frame_count == indices[extracted]:
                    img = frame.to_ndarray(format="rgb24")
                    h, w = img.shape[:2]

                    # Pad to target size
                    pad_h = target_h - h
                    pad_w = target_w - w
                    if pad_h > 0 or pad_w > 0:
                        padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                        padded[:h, :w] = img
                        img = padded
                    elif h > target_h or w > target_w:
                        img = img[:target_h, :target_w]

                    # Convert to float32 CHW [0, 1]
                    img_f = img.astype(np.float32) / 255.0
                    img_f = img_f.transpose(2, 0, 1)  # HWC -> CHW
                    all_frames.append(img_f)
                    extracted += 1
                frame_count += 1

            container.close()
            print(f"  {Path(video_path).name}: {extracted}/{n_frames} frames")

        except Exception as e:
            print(f"  ERROR {Path(video_path).name}: {e}")

    random.shuffle(all_frames)
    print(f"\nTotal: {len(all_frames)} calibration frames")
    return all_frames


def save_calibration_batch(frames_data, output_path):
    """Save frames as flat binary for trtexec --calib."""
    with open(output_path, "wb") as f:
        for frame in frames_data:
            f.write(frame.tobytes())

    size_mb = os.path.getsize(output_path) / 1024**2
    print(f"Saved: {output_path} ({size_mb:.1f} MB, {len(frames_data)} frames)")


def build_int8_engine(onnx_path, calib_data_path, num_calib_frames, output_engine):
    """Build INT8 TRT engine using trtexec."""
    trtexec = str(PROJECT_ROOT / "tools" / "vs" / "vs-plugins" / "vsmlrt-cuda" / "trtexec.exe")
    if not os.path.exists(trtexec):
        raise FileNotFoundError(f"trtexec not found: {trtexec}")

    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        "--shapes=input:1x3x1080x1920",
        "--fp16",
        "--int8",
        f"--calib={calib_data_path}",
        "--calibBatchSize=1",
        "--calibProfile=0",
        "--useCudaGraph",
        f"--saveEngine={output_engine}",
    ]

    print(f"\nBuilding INT8 engine (2-5 min)...")
    print(f"  ONNX: {onnx_path}")
    print(f"  Calibration: {num_calib_frames} frames")
    print(f"  Output: {output_engine}")
    print()

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        print("STDOUT:", result.stdout[-2000:])
        print("STDERR:", result.stderr[-2000:])
        raise RuntimeError(f"trtexec failed with code {result.returncode}")

    for line in result.stdout.split("\n"):
        if any(k in line for k in ["GPU Compute", "Throughput", "PASSED", "FAILED"]):
            print(f"  {line.strip()}")

    size_mb = os.path.getsize(output_engine) / 1024**2
    print(f"\nINT8 engine: {output_engine} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Build INT8 calibration and TRT engine")
    parser.add_argument("--num-frames", type=int, default=200,
                        help="Number of calibration frames (default: 200)")
    parser.add_argument("--onnx", default=str(PROJECT_ROOT / "checkpoints" / "drunet_student" / "drunet_student.onnx"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "checkpoints" / "drunet_student" / "drunet_student_1080p_int8.engine"))
    parser.add_argument("--data-only", action="store_true",
                        help="Only save calibration data, skip engine build")
    args = parser.parse_args()

    frames = extract_random_frames(find_source_videos(), args.num_frames)

    calib_path = str(PROJECT_ROOT / "checkpoints" / "drunet_student" / "calibration_data.bin")
    save_calibration_batch(frames, calib_path)

    if args.data_only:
        print("\n--data-only: Run engine build after GPU is free.")
        return

    build_int8_engine(args.onnx, calib_path, len(frames), args.output)
    print("\nDone! Update ENGINE_PATH in encode_nvencc.vpy to use the INT8 engine.")


if __name__ == "__main__":
    main()
