"""
Remaster smoke test — verify VapourSynth + vs-mlrt + source filter installation.

Runs a series of checks to confirm the pipeline is ready:
1. vspipe binary exists and runs
2. vs-mlrt plugin loads (vstrt.dll)
3. Source filter loads (BestSource, lsmas, or ffms2)
4. encode.vpy loads and can read clip info from a test file
5. (Optional) Single-frame inference to verify model + TRT engine

Usage:
  python remaster/smoke_test.py
  python remaster/smoke_test.py --with-inference   # also test model inference
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def find_vspipe() -> str | None:
    """Find vspipe binary."""
    candidates = [
        PROJECT_ROOT / "tools" / "vs" / "VapourSynth" / "vspipe.exe",
        PROJECT_ROOT / "tools" / "vs" / "vapoursynth" / "vspipe.exe",
        PROJECT_ROOT / "tools" / "vs" / "vspipe.exe",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    # Check PATH
    import shutil
    return shutil.which("vspipe")


def check(name: str, fn) -> bool:
    """Run a check, print result."""
    sys.stdout.write(f"  {name}... ")
    sys.stdout.flush()
    try:
        result = fn()
        if result:
            print(f"OK ({result})" if isinstance(result, str) else "OK")
            return True
        else:
            print("FAIL")
            return False
    except Exception as e:
        print(f"FAIL ({e})")
        return False


def main():
    parser = argparse.ArgumentParser(description="Remaster installation smoke test")
    parser.add_argument("--with-inference", action="store_true",
                        help="Also test single-frame model inference")
    args = parser.parse_args()

    print("Remaster Smoke Test")
    print("=" * 40)
    print()

    all_ok = True

    # 1. Find vspipe
    vspipe = find_vspipe()

    def check_vspipe():
        if vspipe is None:
            raise FileNotFoundError("vspipe not found in tools/vs/ or PATH")
        result = subprocess.run(
            [vspipe, "--version"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            raise RuntimeError(f"vspipe exited with code {result.returncode}")
        # Extract version from output
        for line in (result.stdout + result.stderr).split("\n"):
            if "VapourSynth" in line or "vspipe" in line.lower():
                return line.strip()
        return "found"

    all_ok &= check("vspipe binary", check_vspipe)

    if vspipe is None:
        print("\nCannot continue without vspipe. Run: python tools/setup_vs.py")
        sys.exit(1)

    # 2. Check vs-mlrt plugin
    def check_vstrt():
        # Write a tiny test script
        test_vpy = str(PROJECT_ROOT / "remaster" / "_test_vstrt.vpy")
        with open(test_vpy, "w") as f:
            f.write("""
import vapoursynth as vs
core = vs.core
try:
    v = core.trt.Version()
    trt_ver = v.get("tensorrt_version", b"unknown").decode() if isinstance(v.get("tensorrt_version", b""), bytes) else str(v.get("tensorrt_version", "unknown"))
    # Output a blank clip so vspipe doesn't complain
    clip = core.std.BlankClip(width=16, height=16, length=1, format=vs.RGBS)
    clip.set_output()
except AttributeError:
    raise RuntimeError("vstrt plugin not loaded")
""")
        try:
            result = subprocess.run(
                [vspipe, test_vpy, "-i", "-"],
                capture_output=True, text=True, timeout=30,
            )
            return "loaded" if result.returncode == 0 else None
        finally:
            os.remove(test_vpy)

    all_ok &= check("vs-mlrt (vstrt) plugin", check_vstrt)

    # 3. Check source filter
    def check_source_filter():
        test_vpy = str(PROJECT_ROOT / "remaster" / "_test_source.vpy")
        with open(test_vpy, "w") as f:
            f.write("""
import vapoursynth as vs
core = vs.core
filters = []
try:
    _ = core.bs.VideoSource
    filters.append("BestSource")
except AttributeError:
    pass
try:
    _ = core.lsmas.LWLibavSource
    filters.append("lsmas")
except AttributeError:
    pass
try:
    _ = core.ffms2.Source
    filters.append("ffms2")
except AttributeError:
    pass
if not filters:
    raise RuntimeError("No source filter found")
# Output blank clip
clip = core.std.BlankClip(width=16, height=16, length=1, format=vs.RGBS)
clip.set_output()
""")
        try:
            result = subprocess.run(
                [vspipe, test_vpy, "-i", "-"],
                capture_output=True, text=True, timeout=30,
            )
            return "available" if result.returncode == 0 else None
        finally:
            os.remove(test_vpy)

    all_ok &= check("Source filter (bs/lsmas/ffms2)", check_source_filter)

    # 4. Check encode.vpy loads with a test file
    def check_encode_vpy():
        test_clip = str(PROJECT_ROOT / "data" / "clip_mid_1080p.mp4")
        if not os.path.exists(test_clip):
            return "SKIP (no test clip at data/clip_mid_1080p.mp4)"

        encode_vpy = str(PROJECT_ROOT / "remaster" / "encode.vpy")
        result = subprocess.run(
            [vspipe, encode_vpy, "-a", f"input={test_clip}", "-i", "-"],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0:
            # Parse output for resolution
            for line in result.stdout.split("\n"):
                if "Width" in line or "Height" in line or "Frames" in line:
                    return line.strip()
            return "loaded"
        else:
            stderr_last = result.stderr.strip().split("\n")[-1] if result.stderr else "unknown error"
            raise RuntimeError(stderr_last)

    all_ok &= check("encode.vpy loads test clip", check_encode_vpy)

    # 5. Optional: single-frame inference
    if args.with_inference:
        def check_inference():
            test_clip = str(PROJECT_ROOT / "data" / "clip_mid_1080p.mp4")
            if not os.path.exists(test_clip):
                return "SKIP (no test clip)"

            encode_vpy = str(PROJECT_ROOT / "remaster" / "encode.vpy")
            start = time.perf_counter()
            result = subprocess.run(
                [vspipe, encode_vpy,
                 "-a", f"input={test_clip}",
                 "-c", "y4m",
                 "-s", "0", "-e", "0",  # single frame
                 "-"],
                capture_output=True, timeout=300,  # 5 min for TRT engine build
            )
            elapsed = time.perf_counter() - start

            if result.returncode == 0:
                output_size = len(result.stdout)
                return f"{elapsed:.1f}s, {output_size} bytes output"
            else:
                stderr_last = result.stderr.decode(errors="replace").strip().split("\n")[-1]
                raise RuntimeError(stderr_last)

        all_ok &= check("Single-frame inference", check_inference)

    # Summary
    print()
    if all_ok:
        print("All checks passed. Pipeline is ready.")
        print()
        print("Next steps:")
        print("  python remaster/bench_pipeline.py --vspipe   # benchmark throughput")
        print("  python remaster/encode.py input.mkv output.mkv   # encode a file")
    else:
        print("Some checks failed. Fix issues above, then re-run.")
        print("  python tools/setup_vs.py   # to install missing components")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
