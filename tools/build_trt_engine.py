"""
Build a TensorRT FP16 engine for a specific source resolution.

Engines are static-shape, so each source resolution needs its own. Building one
that matches the source natively is worth it: letterboxed content padded up to
1080p wastes real throughput (1920x816 runs ~55 fps native vs ~43 fps padded).

Usage:
  python tools/build_trt_engine.py --width 1920 --height 816
  python tools/build_trt_engine.py --from-video "E:/plex/movies/some movie.mkv"
  python tools/build_trt_engine.py --list

Output follows the naming convention remaster/vs_remaster.py expects:
  checkpoints/drunet_student/drunet_student_{W}x{H}_fp16.engine
"""
import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRTEXEC = PROJECT_ROOT / "tools" / "vs" / "vs-plugins" / "vsmlrt-cuda" / "trtexec.exe"
ONNX = PROJECT_ROOT / "checkpoints" / "drunet_student" / "drunet_student.onnx"
ENGINE_DIR = PROJECT_ROOT / "checkpoints" / "drunet_student"
FFMPEG = PROJECT_ROOT / "bin" / "ffmpeg.exe"

# DRUNet downsamples 3 times, so both dimensions must be a multiple of 8.
PAD_FACTOR = 8


def probe_resolution(video_path):
    """Read a video's dimensions with ffmpeg (ffprobe isn't bundled)."""
    out = subprocess.run([str(FFMPEG), "-hide_banner", "-i", str(video_path)],
                         capture_output=True, text=True).stderr
    m = re.search(r"Video:.*?[\s,](\d{2,5})x(\d{2,5})[\s,]", out)
    if not m:
        raise RuntimeError(f"Could not determine resolution of {video_path}")
    return int(m.group(1)), int(m.group(2))


def build(width, height, force=False):
    if width % PAD_FACTOR or height % PAD_FACTOR:
        # The .vpy edge-replicate pads up to the engine shape, so round up here.
        width = -(-width // PAD_FACTOR) * PAD_FACTOR
        height = -(-height // PAD_FACTOR) * PAD_FACTOR
        print(f"Rounded up to a multiple of {PAD_FACTOR}: {width}x{height}")

    engine = ENGINE_DIR / f"drunet_student_{width}x{height}_fp16.engine"
    if engine.exists() and not force:
        print(f"Already built: {engine}  (use --force to rebuild)")
        return 0

    log_path = engine.with_suffix(".build.log")
    cmd = [
        str(TRTEXEC),
        f"--onnx={ONNX}",
        f"--shapes=input:1x3x{height}x{width}",
        "--fp16",
        "--inputIOFormats=fp16:chw",
        "--outputIOFormats=fp16:chw",
        "--useCudaGraph",
        f"--saveEngine={engine}",
    ]

    print(f"Building {width}x{height} FP16 engine (a few minutes)...")
    print(f"  Log: {log_path}")
    t0 = time.time()
    with open(log_path, "w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    dt = time.time() - t0

    if rc != 0 or not engine.exists():
        print(f"\nERROR: trtexec failed (rc={rc}). Full log: {log_path}",
              file=sys.stderr)
        return 1

    qps = ""
    for line in log_path.read_text(errors="replace").splitlines():
        if "Throughput:" in line:
            qps = line.split("Throughput:")[1].strip()
    print(f"Built in {dt:.0f}s: {engine.name} ({engine.stat().st_size / 1024**2:.1f} MB)")
    if qps:
        print(f"  Standalone inference: {qps}")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--width", type=int, help="Engine input width")
    p.add_argument("--height", type=int, help="Engine input height")
    p.add_argument("--from-video", help="Take the resolution from this video file")
    p.add_argument("--list", action="store_true", help="List engines already built")
    p.add_argument("--force", action="store_true", help="Rebuild if it exists")
    args = p.parse_args()

    if args.list:
        engines = sorted(ENGINE_DIR.glob("drunet_student_*.engine"))
        if not engines:
            print("No engines built yet.")
        for e in engines:
            print(f"  {e.name}  ({e.stat().st_size / 1024**2:.1f} MB)")
        return 0

    if args.from_video:
        width, height = probe_resolution(args.from_video)
        print(f"{Path(args.from_video).name}: {width}x{height}")
    elif args.width and args.height:
        width, height = args.width, args.height
    else:
        p.error("Give --width and --height, or --from-video, or --list")

    if not TRTEXEC.exists():
        print(f"ERROR: trtexec not found at {TRTEXEC}", file=sys.stderr)
        return 1
    if not ONNX.exists():
        print(f"ERROR: ONNX not found at {ONNX}. Run tools/export_onnx.py first.",
              file=sys.stderr)
        return 1

    return build(width, height, args.force)


if __name__ == "__main__":
    sys.exit(main())
