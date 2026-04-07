"""
Remaster encoding CLI -- wraps vspipe + ffmpeg for batch video enhancement.

Usage:
  python remaster/encode.py input.mkv output.mkv
  python remaster/encode.py input_dir/ output_dir/
  python remaster/encode.py input.mkv output.mkv --cq 20 --preset p5

The heavy lifting happens in C++ (VapourSynth + TensorRT + ffmpeg NVENC).
This script just launches and connects the processes.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Tool discovery
# ---------------------------------------------------------------------------

def find_tool(name: str, extra_paths: list[str] = None) -> str | None:
    """Find an executable by name, checking extra paths first, then PATH."""
    extra_paths = extra_paths or []
    for p in extra_paths:
        candidate = os.path.join(p, name)
        if os.path.isfile(candidate):
            return candidate
        # Try with .exe on Windows
        if sys.platform == "win32" and not name.endswith(".exe"):
            candidate = os.path.join(p, name + ".exe")
            if os.path.isfile(candidate):
                return candidate
    # Fall back to PATH
    found = shutil.which(name)
    return found


def find_vspipe(override: str = None, dry_run: bool = False) -> str:
    """Locate vspipe binary."""
    if override:
        if not dry_run and not os.path.isfile(override):
            raise FileNotFoundError(f"vspipe not found at: {override}")
        return override

    project_root = Path(__file__).resolve().parent.parent
    extra = [
        str(project_root / "tools" / "vs" / "vapoursynth"),
        str(project_root / "tools" / "vs"),
    ]
    found = find_tool("vspipe", extra)
    if found is None:
        raise FileNotFoundError(
            "vspipe not found. Install VapourSynth or specify --vspipe.\n"
            "  Download: https://github.com/vapoursynth/vapoursynth/releases"
        )
    return found


def find_ffmpeg(override: str = None, dry_run: bool = False) -> str:
    """Locate ffmpeg binary."""
    if override:
        if not dry_run and not os.path.isfile(override):
            raise FileNotFoundError(f"ffmpeg not found at: {override}")
        return override

    project_root = Path(__file__).resolve().parent.parent
    extra = [
        str(project_root / "bin"),
    ]
    found = find_tool("ffmpeg", extra)
    if found is None:
        raise FileNotFoundError(
            "ffmpeg not found. Install ffmpeg or specify --ffmpeg.\n"
            "  Download: https://github.com/BtbN/FFmpeg-Builds/releases"
        )
    return found


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_file(
    input_path: str,
    output_path: str,
    vspipe: str,
    ffmpeg: str,
    encode_vpy: str,
    cq: int = 18,
    preset: str = "p4",
    model: str = None,
    dry_run: bool = False,
) -> bool:
    """Encode a single file. Returns True on success."""

    # Build vspipe command
    vspipe_cmd = [
        vspipe,
        encode_vpy,
        "-c", "y4m",
        "-a", f"input={input_path}",
        "-p",  # progress on stderr
        "-",   # output to stdout
    ]
    if model:
        vspipe_cmd.extend(["-a", f"model={model}"])

    # Build ffmpeg command
    # BT.709 color metadata for HD content -- ensures players decode correctly.
    # The y4m from vspipe is already 10-bit YUV420 with correct matrix, but
    # we tag the output so the container carries the right metadata.
    ffmpeg_cmd = [
        ffmpeg,
        "-y",                    # overwrite output
        "-i", "pipe:",           # y4m from stdin
        "-i", input_path,        # original file for audio/subs
        "-map", "0:v:0",         # video from pipe
        "-map", "1:a?",          # audio from original (if exists)
        "-map", "1:s?",          # subtitles from original (if exists)
        "-c:v", "hevc_nvenc",    # NVIDIA hardware encoder
        "-preset", preset,
        "-tune", "hq",
        "-rc", "vbr",
        "-cq", str(cq),
        "-pix_fmt", "p010le",    # 10-bit output
        "-color_primaries", "bt709",
        "-color_trc", "bt709",
        "-colorspace", "bt709",
        "-c:a", "copy",          # passthrough audio
        "-c:s", "copy",          # passthrough subtitles
        output_path,
    ]

    if dry_run:
        print("vspipe command:")
        print("  " + " ".join(vspipe_cmd))
        print()
        print("ffmpeg command:")
        print("  " + " ".join(ffmpeg_cmd))
        return True

    print(f"Encoding: {os.path.basename(input_path)}")
    print(f"  Output: {output_path}")
    print(f"  Quality: cq={cq}, preset={preset}")
    print()

    start = time.time()

    # Launch vspipe -> ffmpeg pipe
    try:
        vspipe_proc = subprocess.Popen(
            vspipe_cmd,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,  # vspipe progress goes to our stderr
        )
        ffmpeg_proc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=vspipe_proc.stdout,
            stdout=subprocess.DEVNULL,
            stderr=sys.stderr,
        )
        # Allow vspipe to receive SIGPIPE if ffmpeg dies
        vspipe_proc.stdout.close()

        ffmpeg_ret = ffmpeg_proc.wait()
        vspipe_ret = vspipe_proc.wait()

    except KeyboardInterrupt:
        print("\nInterrupted -- cleaning up...")
        vspipe_proc.terminate()
        ffmpeg_proc.terminate()
        vspipe_proc.wait()
        ffmpeg_proc.wait()
        # Remove partial output
        if os.path.exists(output_path):
            os.remove(output_path)
        return False

    elapsed = time.time() - start

    if vspipe_ret != 0:
        print(f"\nERROR: vspipe exited with code {vspipe_ret}")
        return False
    if ffmpeg_ret != 0:
        print(f"\nERROR: ffmpeg exited with code {ffmpeg_ret}")
        return False

    # Report results
    output_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
    input_size = os.path.getsize(input_path) if os.path.exists(input_path) else 0
    ratio = output_size / input_size if input_size > 0 else 0

    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Input:  {input_size / 1024 / 1024:.1f} MB")
    print(f"  Output: {output_size / 1024 / 1024:.1f} MB ({ratio:.1%} of original)")
    return True


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------

VIDEO_EXTENSIONS = {".mkv", ".mp4", ".avi", ".m4v", ".ts", ".m2ts", ".webm", ".mov"}


def encode_directory(
    input_dir: str,
    output_dir: str,
    **kwargs,
) -> tuple[int, int]:
    """Encode all video files in a directory. Returns (success, total) counts."""
    os.makedirs(output_dir, exist_ok=True)

    files = sorted([
        f for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS
    ])

    if not files:
        print(f"No video files found in: {input_dir}")
        return 0, 0

    print(f"Found {len(files)} video file(s) in {input_dir}")
    print()

    success = 0
    for i, f in enumerate(files, 1):
        input_path = os.path.join(input_dir, f)
        # Output always .mkv (HEVC container)
        output_name = os.path.splitext(f)[0] + ".mkv"
        output_path = os.path.join(output_dir, output_name)

        if os.path.exists(output_path):
            print(f"[{i}/{len(files)}] SKIP (exists): {f}")
            success += 1
            continue

        print(f"[{i}/{len(files)}] ", end="")
        if encode_file(input_path, output_path, **kwargs):
            success += 1
        print()

    return success, len(files)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Remaster -- enhance video with DRUNet via VapourSynth + TensorRT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python remaster/encode.py input.mkv output.mkv
  python remaster/encode.py input.mkv output.mkv --cq 20 --preset p5
  python remaster/encode.py /videos/input/ /videos/output/
  python remaster/encode.py input.mkv output.mkv --dry-run
""",
    )
    parser.add_argument("input", help="Input video file or directory")
    parser.add_argument("output", help="Output video file or directory")
    parser.add_argument("--cq", type=int, default=18,
                        help="Constant quality (default: 18, lower = higher quality)")
    parser.add_argument("--preset", default="p4",
                        help="NVENC preset: p1 (fastest) to p7 (best quality). Default: p4")
    parser.add_argument("--model", default=None,
                        help="Path to ONNX model (default: auto-detect)")
    parser.add_argument("--vspipe", default=None,
                        help="Path to vspipe binary (default: auto-detect)")
    parser.add_argument("--ffmpeg", default=None,
                        help="Path to ffmpeg binary (default: auto-detect)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")

    args = parser.parse_args()

    # Resolve tools
    try:
        vspipe = find_vspipe(args.vspipe, dry_run=args.dry_run)
        ffmpeg = find_ffmpeg(args.ffmpeg, dry_run=args.dry_run)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # Resolve encode.vpy path
    encode_vpy = str(Path(__file__).resolve().parent / "encode.vpy")
    if not os.path.exists(encode_vpy):
        print(f"ERROR: encode.vpy not found at {encode_vpy}", file=sys.stderr)
        sys.exit(1)

    print(f"Tools: vspipe={vspipe}")
    print(f"       ffmpeg={ffmpeg}")
    print(f"       script={encode_vpy}")
    print()

    common_kwargs = dict(
        vspipe=vspipe,
        ffmpeg=ffmpeg,
        encode_vpy=encode_vpy,
        cq=args.cq,
        preset=args.preset,
        model=args.model,
        dry_run=args.dry_run,
    )

    # Dispatch: single file or directory
    if os.path.isdir(args.input):
        success, total = encode_directory(args.input, args.output, **common_kwargs)
        print(f"Batch complete: {success}/{total} succeeded")
        sys.exit(0 if success == total else 1)
    else:
        if not os.path.isfile(args.input):
            print(f"ERROR: Input not found: {args.input}", file=sys.stderr)
            sys.exit(1)
        ok = encode_file(args.input, args.output, **common_kwargs)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
