"""
Remaster encoding CLI -- NVEncC + VapourSynth + TensorRT.

Fastest encoding path: VapourSynth runs in-process with NVEncC (no pipe).
Audio is copied from the original file.

Usage:
  python remaster/encode_nvencc.py input.mkv output.mkv
  python remaster/encode_nvencc.py input.mkv output.mkv --cq 20
  python remaster/encode_nvencc.py input_dir/ output_dir/
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NVENCC = str(PROJECT_ROOT / "bin" / "NVEncC" / "NVEncC64.exe")
VS_DIR = str(PROJECT_ROOT / "tools" / "vs")
ENCODE_VPY = str(PROJECT_ROOT / "remaster" / "encode_nvencc.vpy")

VIDEO_EXTENSIONS = {".mkv", ".mp4", ".avi", ".m4v", ".ts", ".m2ts", ".webm", ".mov"}


def encode_file(input_path, output_path, cq=18, preset="p4"):
    """Encode a single file with NVEncC + VapourSynth TRT."""
    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    env = os.environ.copy()
    env["REMASTER_INPUT"] = input_path
    env["PATH"] = VS_DIR + os.pathsep + env.get("PATH", "")

    cmd = [
        NVENCC,
        "--vpy", "--vpy-mt",
        "--vsdir", VS_DIR,
        "-i", ENCODE_VPY,
        "--codec", "hevc",
        "--profile", "main10",
        "--output-depth", "10",
        "--preset", preset,
        "--vbr-quality", str(cq),
        "--audio-source", f"{input_path}:copy",  # copy audio from original
        "--colormatrix", "bt709",
        "--colorprim", "bt709",
        "--transfer", "bt709",
        "-o", output_path,
    ]

    print(f"Encoding: {os.path.basename(input_path)}")
    print(f"  Output: {output_path}")
    print(f"  Quality: cq={cq}, preset={preset}")
    print()

    start = time.time()
    result = subprocess.run(cmd, env=env)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\nERROR: NVEncC exited with code {result.returncode}")
        return False

    if os.path.exists(output_path):
        out_size = os.path.getsize(output_path) / 1024**2
        in_size = os.path.getsize(input_path) / 1024**2
        print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f} min)")
        print(f"  Input:  {in_size:.0f} MB")
        print(f"  Output: {out_size:.0f} MB")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Remaster -- NVEncC + VapourSynth + TensorRT (fastest path)")
    parser.add_argument("input", help="Input video file or directory")
    parser.add_argument("output", help="Output video file or directory")
    parser.add_argument("--cq", type=int, default=18,
                        help="Constant quality (default: 18)")
    parser.add_argument("--preset", default="p4",
                        help="NVENC preset: p1-p7 (default: p4)")
    args = parser.parse_args()

    if not os.path.exists(NVENCC):
        print(f"ERROR: NVEncC not found at {NVENCC}", file=sys.stderr)
        sys.exit(1)

    if os.path.isdir(args.input):
        os.makedirs(args.output, exist_ok=True)
        files = sorted(f for f in os.listdir(args.input)
                       if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS)
        print(f"Batch: {len(files)} files")
        ok = 0
        for i, f in enumerate(files, 1):
            out_name = os.path.splitext(f)[0] + ".mkv"
            out_path = os.path.join(args.output, out_name)
            if os.path.exists(out_path):
                print(f"[{i}/{len(files)}] SKIP: {f}")
                ok += 1
                continue
            print(f"\n[{i}/{len(files)}]")
            if encode_file(os.path.join(args.input, f), out_path, args.cq, args.preset):
                ok += 1
        print(f"\nBatch: {ok}/{len(files)} succeeded")
    else:
        if not encode_file(args.input, args.output, args.cq, args.preset):
            sys.exit(1)


if __name__ == "__main__":
    main()
