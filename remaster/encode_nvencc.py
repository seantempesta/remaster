"""
Remaster encoding CLI -- NVEncC + VapourSynth + TensorRT.

Fastest encoding path: VapourSynth runs in-process with NVEncC (no pipe).
Audio is copied from the original file.

Usage:
  python remaster/encode_nvencc.py input.mkv output.mkv
  python remaster/encode_nvencc.py input.mkv output.mkv --cq 20
  python remaster/encode_nvencc.py input_dir/ output_dir/
  python remaster/encode_nvencc.py input.mkv preview.mkv --trim 28800 300
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import build_trt_engine  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NVENCC = str(PROJECT_ROOT / "bin" / "NVEncC" / "NVEncC64.exe")
VS_DIR = str(PROJECT_ROOT / "tools" / "vs")
ENCODE_VPY = str(PROJECT_ROOT / "remaster" / "encode_nvencc.vpy")

VIDEO_EXTENSIONS = {".mkv", ".mp4", ".avi", ".m4v", ".ts", ".m2ts", ".webm", ".mov"}


def ensure_engine(input_path):
    """Build a native-resolution engine for this source if one doesn't exist.

    Engines are static-shape. Without this the first run on any new resolution
    dies with a FileNotFoundError telling the user to go run another command --
    so just build it, the same way the Docker entrypoint does.
    """
    width, height = build_trt_engine.probe_resolution(input_path)
    # Engines are built at the source size rounded up to a multiple of 8, so
    # check for that name -- otherwise this rebuilds on every single run.
    ew = build_trt_engine.round_up(width)
    eh = build_trt_engine.round_up(height)
    engine = (PROJECT_ROOT / "checkpoints" / "drunet_student" /
              f"drunet_student_{ew}x{eh}_fp16.engine")
    if engine.exists():
        return
    print(f"No engine for {ew}x{eh} yet -- building one (a few minutes,\n"
          f"one-time per resolution). Native beats padding up to 1080p on speed.")
    if build_trt_engine.build(width, height) != 0:
        raise RuntimeError(f"Could not build a TRT engine for {width}x{height}")


def encode_file(input_path, output_path, cq=18, preset="p4", engine=None, trim=None):
    """Encode a single file with NVEncC + VapourSynth TRT."""
    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    env = os.environ.copy()
    env["REMASTER_INPUT"] = input_path
    if trim:
        # NVEncC's --seek is silently IGNORED for --vpy input, so a spot-check
        # that asks to start at 20:00 actually encodes from frame 0 -- which on
        # many releases is a black lead-in and looks like a broken pipeline.
        # Trimming inside the script is the only thing that actually works.
        env["REMASTER_TRIM_START"] = str(trim[0])
        env["REMASTER_TRIM_LENGTH"] = str(trim[1])
    # NVEncC evaluates the .vpy with its own cwd, so tell it where to import
    # vs_remaster.py from rather than relying on __file__/cwd guesswork.
    env["REMASTER_DIR"] = str(PROJECT_ROOT / "remaster")
    if engine:
        env["REMASTER_ENGINE"] = os.path.abspath(engine)
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
        "--colormatrix", "bt709",
        "--colorprim", "bt709",
        "--transfer", "bt709",
    ]

    # A trimmed preview is a few seconds of video; muxing the full-length audio
    # and subtitle tracks against it just produces a desynced mess.
    if not trim:
        cmd += [
            "--audio-source", f"{input_path}:copy",  # copy audio from original
            "--sub-source", f"{input_path}",         # carry subtitle tracks through
        ]

    cmd += ["-o", output_path]

    print(f"Encoding: {os.path.basename(input_path)}")
    print(f"  Output: {output_path}")
    print(f"  Quality: cq={cq}, preset={preset}")
    if trim:
        print(f"  Preview: {trim[1]} frames from frame {trim[0]} (no audio/subs)")
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
    parser.add_argument("--engine",
                        help="TRT engine to use (default: auto-select by source "
                             "resolution, building one if needed)")
    parser.add_argument("--trim", nargs=2, type=int, metavar=("START", "LENGTH"),
                        help="Encode only LENGTH frames starting at frame START, "
                             "with no audio or subtitles. Use this for spot checks "
                             "-- NVEncC's own --seek does nothing with .vpy input.")
    args = parser.parse_args()

    if not os.path.exists(NVENCC):
        print(f"ERROR: NVEncC not found at {NVENCC}", file=sys.stderr)
        sys.exit(1)

    if args.trim and os.path.isdir(args.input):
        parser.error("--trim works on a single file, not a directory")

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
            src = os.path.join(args.input, f)
            if not args.engine:
                ensure_engine(src)
            if encode_file(src, out_path, args.cq, args.preset, args.engine):
                ok += 1
        print(f"\nBatch: {ok}/{len(files)} succeeded")
    else:
        if not args.engine:
            ensure_engine(args.input)
        if not encode_file(args.input, args.output, args.cq, args.preset,
                           args.engine, args.trim):
            sys.exit(1)


if __name__ == "__main__":
    main()
