"""
Remaster pipeline throughput benchmark.

Tests each pipeline component independently with synthetic data to find
bottlenecks and verify we're hitting hardware ceilings.

Usage:
  python remaster/bench_pipeline.py              # run all benchmarks
  python remaster/bench_pipeline.py --nvenc      # NVENC encode only
  python remaster/bench_pipeline.py --pipe       # pipe bandwidth only
  python remaster/bench_pipeline.py --vspipe     # VapourSynth throughput only

Each test generates synthetic data to isolate the component from upstream/
downstream stages. Results are compared against theoretical maximums.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FFMPEG = str(PROJECT_ROOT / "bin" / "ffmpeg.exe")
WIDTH, HEIGHT = 1920, 1080
# 10-bit YUV420: Y=2*W*H, U=V=2*(W/2)*(H/2)
Y_SIZE = WIDTH * HEIGHT * 2
UV_SIZE = (WIDTH // 2) * (HEIGHT // 2) * 2
FRAME_SIZE = Y_SIZE + 2 * UV_SIZE
Y4M_HEADER = f"YUV4MPEG2 W{WIDTH} H{HEIGHT} F24000:1001 Ip A1:1 C420p10\n".encode()


def generate_frame_data() -> bytes:
    """Generate a single frame of synthetic data (deterministic pattern)."""
    return bytes([(j * 37 + 7) % 256 for j in range(FRAME_SIZE)])


# ---------------------------------------------------------------------------
# NVENC encode benchmark
# ---------------------------------------------------------------------------

def bench_nvenc(frames: int = 300):
    """Benchmark NVENC HEVC encoding throughput at various presets."""
    print(f"=== NVENC Encode Benchmark ({WIDTH}x{HEIGHT}, {frames} frames) ===")
    print(f"Frame size: {FRAME_SIZE / 1024 / 1024:.1f} MB (YUV420P10)")
    print()

    if not os.path.exists(FFMPEG):
        print(f"ERROR: ffmpeg not found at {FFMPEG}")
        return

    frame_data = generate_frame_data()

    for preset in ["p1", "p4", "p7"]:
        output = str(PROJECT_ROOT / "data" / f"bench_nvenc_{preset}.mkv")
        cmd = [
            FFMPEG, "-y", "-f", "yuv4mpegpipe", "-i", "pipe:",
            "-c:v", "hevc_nvenc", "-preset", preset,
            "-tune", "hq", "-rc", "vbr", "-cq", "18",
            "-pix_fmt", "p010le", output,
        ]

        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        proc.stdin.write(Y4M_HEADER)

        start = time.perf_counter()
        for _ in range(frames):
            proc.stdin.write(b"FRAME\n")
            proc.stdin.write(frame_data)
        proc.stdin.close()
        proc.wait()
        elapsed = time.perf_counter() - start

        fps = frames / elapsed
        latency_ms = elapsed / frames * 1000
        size_mb = os.path.getsize(output) / 1024 / 1024 if os.path.exists(output) else 0
        bw_gbps = frames * FRAME_SIZE / elapsed / 1e9

        status = "OK" if fps >= 78 else "BOTTLENECK"
        print(f"  preset {preset}: {fps:6.1f} fps  {latency_ms:5.1f} ms/frame  "
              f"{bw_gbps:.2f} GB/s  {size_mb:5.1f} MB  [{status}]")

        if os.path.exists(output):
            os.remove(output)

    print()


# ---------------------------------------------------------------------------
# Pipe bandwidth benchmark
# ---------------------------------------------------------------------------

def bench_pipe(frames: int = 500):
    """Benchmark y4m pipe throughput (Python -> ffmpeg, no encode)."""
    print(f"=== Pipe Bandwidth Benchmark ({WIDTH}x{HEIGHT}, {frames} frames) ===")
    print()

    if not os.path.exists(FFMPEG):
        print(f"ERROR: ffmpeg not found at {FFMPEG}")
        return

    frame_data = generate_frame_data()

    # Pipe to ffmpeg null output
    cmd = [FFMPEG, "-y", "-f", "yuv4mpegpipe", "-i", "pipe:", "-f", "null", "-"]
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    proc.stdin.write(Y4M_HEADER)

    start = time.perf_counter()
    for _ in range(frames):
        proc.stdin.write(b"FRAME\n")
        proc.stdin.write(frame_data)
    proc.stdin.close()
    proc.wait()
    elapsed = time.perf_counter() - start

    fps = frames / elapsed
    bw_gbps = frames * FRAME_SIZE / elapsed / 1e9
    needed_gbps = 78 * FRAME_SIZE / 1e9
    headroom = bw_gbps / needed_gbps

    print(f"  Pipe -> ffmpeg null: {fps:.1f} fps ({bw_gbps:.2f} GB/s)")
    print(f"  Needed at 78 fps:   {needed_gbps:.2f} GB/s")
    print(f"  Headroom:           {headroom:.1f}x")

    status = "OK" if headroom >= 1.5 else "TIGHT" if headroom >= 1.0 else "BOTTLENECK"
    print(f"  Status:             [{status}]")
    print()


# ---------------------------------------------------------------------------
# VapourSynth benchmark (requires VS + vs-mlrt installed)
# ---------------------------------------------------------------------------

def bench_vspipe():
    """Benchmark VapourSynth + vs-mlrt throughput with a real video."""
    print(f"=== VapourSynth Pipeline Benchmark ===")
    print()

    # Find vspipe
    vspipe = None
    candidates = [
        str(PROJECT_ROOT / "tools" / "vs" / "vapoursynth" / "vspipe.exe"),
        str(PROJECT_ROOT / "tools" / "vs" / "vspipe.exe"),
    ]
    for c in candidates:
        if os.path.exists(c):
            vspipe = c
            break
    if vspipe is None:
        from shutil import which
        vspipe = which("vspipe")
    if vspipe is None:
        print("  SKIP: vspipe not found. Install VapourSynth to run this benchmark.")
        print()
        return

    # Find a test clip
    test_clip = str(PROJECT_ROOT / "data" / "clip_mid_1080p.mp4")
    if not os.path.exists(test_clip):
        print(f"  SKIP: test clip not found at {test_clip}")
        print()
        return

    encode_vpy = str(PROJECT_ROOT / "remaster" / "encode.vpy")

    # Test 1: vspipe to null (inference throughput only)
    print("  Test 1: vspipe -> null (inference throughput)")
    cmd = [
        vspipe, encode_vpy,
        "-a", f"input={test_clip}",
        "-p",  # progress
        "-",   # stdout
    ]

    start = time.perf_counter()
    proc = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    _, stderr = proc.communicate()
    elapsed = time.perf_counter() - start

    stderr_text = stderr.decode(errors="replace")
    # Parse frame count from vspipe progress output
    # Format: "Frame: N/Total (X.XX fps)"
    import re
    fps_match = re.search(r"([\d.]+) fps", stderr_text)
    frame_match = re.search(r"Frame: (\d+)/(\d+)", stderr_text)

    if fps_match:
        reported_fps = float(fps_match.group(1))
        print(f"    Reported: {reported_fps:.1f} fps")
    if frame_match:
        frames_done = int(frame_match.group(1))
        total_frames = int(frame_match.group(2))
        measured_fps = frames_done / elapsed
        print(f"    Measured: {measured_fps:.1f} fps ({frames_done}/{total_frames} frames in {elapsed:.1f}s)")

    if proc.returncode != 0:
        print(f"    ERROR: vspipe exited with code {proc.returncode}")
        # Print last few lines of stderr for debugging
        for line in stderr_text.strip().split("\n")[-5:]:
            print(f"    {line}")

    # Test 2: vspipe -> ffmpeg NVENC (full pipeline)
    print()
    print("  Test 2: vspipe -> ffmpeg NVENC (full pipeline)")
    output = str(PROJECT_ROOT / "data" / "bench_vspipe_nvenc.mkv")

    vspipe_cmd = [
        vspipe, encode_vpy,
        "-c", "y4m",
        "-a", f"input={test_clip}",
        "-p",
        "-",
    ]
    ffmpeg_cmd = [
        FFMPEG, "-y",
        "-f", "yuv4mpegpipe", "-i", "pipe:",
        "-c:v", "hevc_nvenc", "-preset", "p4",
        "-tune", "hq", "-rc", "vbr", "-cq", "18",
        "-pix_fmt", "p010le", output,
    ]

    start = time.perf_counter()
    vspipe_proc = subprocess.Popen(
        vspipe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    ffmpeg_proc = subprocess.Popen(
        ffmpeg_cmd, stdin=vspipe_proc.stdout,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    vspipe_proc.stdout.close()

    ffmpeg_proc.wait()
    vspipe_proc.wait()
    elapsed = time.perf_counter() - start

    vs_stderr = vspipe_proc.stderr.read().decode(errors="replace")
    fps_match = re.search(r"([\d.]+) fps", vs_stderr)

    if os.path.exists(output):
        size_mb = os.path.getsize(output) / 1024 / 1024
        print(f"    Output: {size_mb:.1f} MB")
        os.remove(output)

    if fps_match:
        reported_fps = float(fps_match.group(1))
        print(f"    Reported: {reported_fps:.1f} fps ({elapsed:.1f}s)")
        status = "TARGET MET" if reported_fps >= 72 else "BELOW TARGET" if reported_fps >= 40 else "INVESTIGATE"
        print(f"    Status: [{status}] (target: 72+ fps)")
    elif vspipe_proc.returncode != 0:
        print(f"    ERROR: vspipe exited with code {vspipe_proc.returncode}")
        for line in vs_stderr.strip().split("\n")[-5:]:
            print(f"    {line}")

    print()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary():
    """Print theoretical maximums and pipeline budget."""
    print("=== Pipeline Throughput Budget (RTX 3060, 1080p) ===")
    print()
    print("  Component          | Measured     | Ceiling       | Status")
    print("  -------------------|--------------|---------------|--------")
    print("  NAFNet inference    | 78 fps       | ~85-100*      | NEAR CEILING")
    print("  NVDEC HEVC decode  | untested**   | ~60 fps       | POSSIBLE LIMIT")
    print("  NVENC p4 encode    | 196-221 fps  | ~220 fps      | OK (2.5x headroom)")
    print("  NVENC p7 encode    | 122-124 fps  | ~125 fps      | OK (1.6x headroom)")
    print("  y4m pipe bandwidth | 761 fps      | ~800+ fps     | OK (9.7x headroom)")
    print("  VapourSynth sched  | untested     | ~1000+ fps*** | OK")
    print()
    print("  *   TensorRT kernel fusion reduces intermediate memory traffic.")
    print("        Model is memory-bandwidth-bound (16 GB/s of 336 GB/s).")
    print("  **  NVDEC spec for HEVC 1080p on Ampere. BestSource uses CPU")
    print("        decode by default; NVDEC available via hwdevice='cuda'.")
    print("  *** C++ mutex try_lock is ~0.5us/frame, negligible at 78 fps.")
    print()
    print("  Conclusion: Model inference at 78 fps IS the ceiling.")
    print("  NVDEC at ~60 fps could become the limit if inference is faster.")
    print("  TensorRT may push inference to 85-100 fps via kernel fusion.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Remaster pipeline throughput benchmark")
    parser.add_argument("--nvenc", action="store_true", help="NVENC encode benchmark only")
    parser.add_argument("--pipe", action="store_true", help="Pipe bandwidth benchmark only")
    parser.add_argument("--vspipe", action="store_true", help="VapourSynth benchmark only")
    parser.add_argument("--summary", action="store_true", help="Print throughput budget only")
    args = parser.parse_args()

    run_all = not (args.nvenc or args.pipe or args.vspipe or args.summary)

    if run_all or args.nvenc:
        bench_nvenc()
    if run_all or args.pipe:
        bench_pipe()
    if run_all or args.vspipe:
        bench_vspipe()
    if run_all or args.summary:
        print_summary()


if __name__ == "__main__":
    main()
