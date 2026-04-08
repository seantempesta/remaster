"""
Setup script for VapourSynth + vs-mlrt + source filters on Windows.

Installs everything needed for the remaster pipeline:
  - VapourSynth portable (vspipe.exe)
  - vs-mlrt with TensorRT backend (vstrt.dll + vsmlrt.py)
  - L-SMASH-Works source filter (lsmas.dll)

Usage:
  python tools/setup_vs.py                 # install all components
  python tools/setup_vs.py --build-engine  # also build TensorRT engine

All components install to tools/vs/ under the project root.
"""

import argparse
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

# ---------------------------------------------------------------------------
# Configuration — pin specific versions for reproducibility
# ---------------------------------------------------------------------------

# VapourSynth R70 portable (Python 3.12 embedded)
# https://github.com/vapoursynth/vapoursynth/releases
VS_VERSION = "R70"
VS_URL = (
    "https://github.com/vapoursynth/vapoursynth/releases/download/{version}/"
    "VapourSynth64-Portable-{version}.zip"
).format(version=VS_VERSION)
VS_SHA256 = None  # TODO: fill in after first successful download

# vs-mlrt v16.2 (TensorRT 10.x backend for VapourSynth)
# https://github.com/AmusementClub/vs-mlrt/releases
# The release zip contains:
#   vstrt.dll           - VapourSynth plugin
#   vsmlrt.py           - Python helper (goes on sys.path)
#   TensorRT runtime DLLs (nvinfer.dll, etc.)
VSMLRT_VERSION = "v16.2"
VSMLRT_URL = (
    "https://github.com/AmusementClub/vs-mlrt/releases/download/{version}/"
    "vsmlrt-windows-x64-cuda.zip"
).format(version=VSMLRT_VERSION)
VSMLRT_SHA256 = None  # TODO: fill in after first successful download

# L-SMASH-Works (source filter for decoding video in VapourSynth)
# https://github.com/HomeOfAviSynthPlusEvolution/L-SMASH-Works/releases
# Provides: core.lsmas.LWLibavSource() and core.lsmas.LibavSMASHSource()
LSMASH_VERSION = "20240825"
LSMASH_URL = (
    "https://github.com/HomeOfAviSynthPlusEvolution/L-SMASH-Works/releases/download/"
    "{version}/L-SMASH-Works_{version}.zip"
).format(version=LSMASH_VERSION)
LSMASH_SHA256 = None  # TODO: fill in after first successful download

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
VS_ROOT = os.path.join(SCRIPT_DIR, "vs")
DOWNLOADS_DIR = os.path.join(VS_ROOT, "_downloads")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[setup] {msg}")


def log_error(msg: str) -> None:
    print(f"[setup] ERROR: {msg}", file=sys.stderr)


def check_sha256(path: str, expected: str | None) -> bool:
    """Verify SHA256 hash. Returns True if matches or no hash provided."""
    if expected is None:
        return True
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != expected:
        log_error(f"SHA256 mismatch for {os.path.basename(path)}")
        log_error(f"  expected: {expected}")
        log_error(f"  actual:   {actual}")
        return False
    return True


def download_file(url: str, dest: str, sha256: str | None = None) -> str:
    """
    Download a file with resume support and progress display.
    Returns the path to the downloaded file.
    """
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    # Check if already downloaded and valid
    if os.path.exists(dest):
        if check_sha256(dest, sha256):
            log(f"Already downloaded: {os.path.basename(dest)}")
            return dest
        else:
            log(f"Existing file failed hash check, re-downloading")
            os.remove(dest)

    # Support resuming partial downloads
    partial = dest + ".partial"
    existing_size = 0
    if os.path.exists(partial):
        existing_size = os.path.getsize(partial)

    log(f"Downloading {os.path.basename(dest)}...")
    log(f"  URL: {url}")

    req = urllib.request.Request(url)
    if existing_size > 0:
        req.add_header("Range", f"bytes={existing_size}-")
        log(f"  Resuming from {existing_size / 1024 / 1024:.1f} MB")

    try:
        resp = urllib.request.urlopen(req)
    except urllib.error.HTTPError as e:
        if e.code == 416:
            # Range not satisfiable — file is complete
            os.rename(partial, dest)
            return dest
        raise

    # Get total size from Content-Range or Content-Length
    total = None
    content_range = resp.headers.get("Content-Range")
    if content_range:
        total = int(content_range.split("/")[-1])
    elif resp.headers.get("Content-Length"):
        total = existing_size + int(resp.headers["Content-Length"])

    mode = "ab" if existing_size > 0 else "wb"
    downloaded = existing_size

    with open(partial, mode) as f:
        while True:
            chunk = resp.read(1 << 18)  # 256 KB chunks
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                mb = downloaded / 1024 / 1024
                total_mb = total / 1024 / 1024
                print(f"\r  {mb:.1f} / {total_mb:.1f} MB ({pct:.0f}%)", end="", flush=True)
            else:
                mb = downloaded / 1024 / 1024
                print(f"\r  {mb:.1f} MB", end="", flush=True)
    print()  # newline after progress

    os.rename(partial, dest)

    if not check_sha256(dest, sha256):
        log_error("Download hash mismatch — file may be corrupt")
        sys.exit(1)

    return dest


def extract_zip(zip_path: str, dest_dir: str) -> None:
    """Extract a zip file, showing what's happening."""
    log(f"Extracting {os.path.basename(zip_path)} -> {dest_dir}")
    os.makedirs(dest_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)


def find_files_by_ext(directory: str, ext: str) -> list[str]:
    """Recursively find files with a given extension."""
    results = []
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(ext.lower()):
                results.append(os.path.join(root, f))
    return results


def disk_space_gb(path: str) -> float:
    """Return available disk space in GB for the drive containing path."""
    total, used, free = shutil.disk_usage(os.path.splitdrive(path)[0] or path)
    return free / (1024 ** 3)


# ---------------------------------------------------------------------------
# Prerequisite checks
# ---------------------------------------------------------------------------

def check_prerequisites() -> bool:
    """Check that all prerequisites are met. Returns True if OK."""
    ok = True

    # Python version
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 10):
        log_error(f"Python 3.10+ required, found {major}.{minor}")
        ok = False
    else:
        log(f"Python {major}.{minor} - OK")

    # Windows
    if sys.platform != "win32":
        log_error("This script is designed for Windows")
        ok = False
    else:
        log(f"Platform: Windows ({platform.version()}) - OK")

    # NVIDIA GPU
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            gpu_info = result.stdout.strip().split("\n")[0]
            log(f"GPU: {gpu_info} - OK")
        else:
            log_error("nvidia-smi failed — is an NVIDIA GPU installed?")
            ok = False
    except FileNotFoundError:
        log_error("nvidia-smi not found — NVIDIA drivers not installed")
        ok = False
    except subprocess.TimeoutExpired:
        log_error("nvidia-smi timed out")
        ok = False

    # Disk space
    free_gb = disk_space_gb(VS_ROOT if os.path.exists(VS_ROOT) else PROJECT_ROOT)
    if free_gb < 2.0:
        log_error(f"Need ~2 GB free disk space, only {free_gb:.1f} GB available")
        ok = False
    else:
        log(f"Disk space: {free_gb:.1f} GB free - OK")

    return ok


# ---------------------------------------------------------------------------
# Install steps
# ---------------------------------------------------------------------------

def install_vapoursynth() -> bool:
    """Download and extract VapourSynth portable."""
    vspipe = os.path.join(VS_ROOT, "VSPipe.exe")
    if os.path.isfile(vspipe):
        log(f"VapourSynth already installed at {VS_ROOT}")
        return True

    zip_name = os.path.basename(VS_URL)
    zip_path = os.path.join(DOWNLOADS_DIR, zip_name)

    download_file(VS_URL, zip_path, VS_SHA256)

    # Extract directly into VS_ROOT (zip contains files at top level)
    # Use a temp dir first so we can handle nested folder structure
    with tempfile.TemporaryDirectory() as tmp:
        extract_zip(zip_path, tmp)

        # Check if contents are in a subfolder or at top level
        entries = os.listdir(tmp)
        if len(entries) == 1 and os.path.isdir(os.path.join(tmp, entries[0])):
            src = os.path.join(tmp, entries[0])
        else:
            src = tmp

        # Copy files to VS_ROOT
        os.makedirs(VS_ROOT, exist_ok=True)
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(VS_ROOT, item)
            if os.path.exists(d):
                continue  # don't overwrite existing files
            if os.path.isdir(s):
                shutil.copytree(s, d)
            else:
                shutil.copy2(s, d)

    if os.path.isfile(vspipe):
        log(f"VapourSynth {VS_VERSION} installed successfully")
        return True
    else:
        log_error("VSPipe.exe not found after extraction — zip layout may have changed")
        log_error(f"Check contents of {zip_path} and update the script")
        return False


def install_vsmlrt() -> bool:
    """Download and install vs-mlrt (TensorRT backend for VapourSynth)."""
    plugins_dir = os.path.join(VS_ROOT, "vs-plugins")
    os.makedirs(plugins_dir, exist_ok=True)

    # Check if already installed
    vstrt_dll = os.path.join(plugins_dir, "vstrt.dll")
    if os.path.isfile(vstrt_dll):
        log("vs-mlrt already installed")
        return True

    zip_name = os.path.basename(VSMLRT_URL)
    zip_path = os.path.join(DOWNLOADS_DIR, zip_name)

    download_file(VSMLRT_URL, zip_path, VSMLRT_SHA256)

    with tempfile.TemporaryDirectory() as tmp:
        extract_zip(zip_path, tmp)

        # Find and copy vstrt.dll to vs-plugins
        dll_files = find_files_by_ext(tmp, ".dll")
        py_files = find_files_by_ext(tmp, ".py")

        # Copy all DLLs to vs-plugins (vstrt.dll + TensorRT runtime DLLs)
        dll_count = 0
        for dll in dll_files:
            dest = os.path.join(plugins_dir, os.path.basename(dll))
            if not os.path.exists(dest):
                shutil.copy2(dll, dest)
                dll_count += 1

        # Copy vsmlrt.py to a known location for import
        # The .vpy scripts look in reference-code/vs-mlrt/scripts/
        vsmlrt_scripts_dir = os.path.join(
            PROJECT_ROOT, "reference-code", "vs-mlrt", "scripts"
        )
        os.makedirs(vsmlrt_scripts_dir, exist_ok=True)

        for py in py_files:
            if os.path.basename(py) == "vsmlrt.py":
                dest = os.path.join(vsmlrt_scripts_dir, "vsmlrt.py")
                shutil.copy2(py, dest)
                log(f"Copied vsmlrt.py to {vsmlrt_scripts_dir}")
                break

        log(f"Installed {dll_count} DLL(s) to {plugins_dir}")

    if os.path.isfile(vstrt_dll):
        log(f"vs-mlrt {VSMLRT_VERSION} installed successfully")
        return True
    else:
        log_error("vstrt.dll not found after extraction — release layout may differ")
        log_error(f"Manually extract {zip_path} and copy vstrt.dll to {plugins_dir}")
        return False


def install_lsmash() -> bool:
    """Download and install L-SMASH-Works source filter."""
    plugins_dir = os.path.join(VS_ROOT, "vs-plugins")
    os.makedirs(plugins_dir, exist_ok=True)

    # Check if already installed (any of the known source filters)
    for name in ["vslsmashsource.dll", "LSMASHSource.dll"]:
        if os.path.isfile(os.path.join(plugins_dir, name)):
            log("L-SMASH-Works already installed")
            return True

    zip_name = os.path.basename(LSMASH_URL)
    zip_path = os.path.join(DOWNLOADS_DIR, zip_name)

    download_file(LSMASH_URL, zip_path, LSMASH_SHA256)

    with tempfile.TemporaryDirectory() as tmp:
        extract_zip(zip_path, tmp)

        # Find the 64-bit DLL (may be in x64/ subfolder)
        dll_files = find_files_by_ext(tmp, ".dll")

        installed = False
        for dll in dll_files:
            basename = os.path.basename(dll).lower()
            # Look for the VapourSynth plugin DLL (not AviSynth)
            # Typical names: vslsmashsource.dll, LSMASHSource.dll
            if "lsmash" in basename or "lwlibav" in basename:
                # Prefer x64 over x86 if both exist
                parent = os.path.basename(os.path.dirname(dll)).lower()
                if "x86" in parent or "32" in parent:
                    continue
                dest = os.path.join(plugins_dir, os.path.basename(dll))
                if not os.path.exists(dest):
                    shutil.copy2(dll, dest)
                    log(f"Installed {os.path.basename(dll)}")
                installed = True

        if installed:
            log(f"L-SMASH-Works {LSMASH_VERSION} installed successfully")
            return True
        else:
            log_error("Could not find L-SMASH-Works DLL in the release zip")
            log_error(f"Manually extract {zip_path} and copy the DLL to {plugins_dir}")
            return False


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_vspipe() -> bool:
    """Run vspipe --version to verify the installation."""
    vspipe = os.path.join(VS_ROOT, "VSPipe.exe")
    if not os.path.isfile(vspipe):
        log_error("VSPipe.exe not found")
        return False

    try:
        result = subprocess.run(
            [vspipe, "--version"],
            capture_output=True, text=True, timeout=15,
            env={**os.environ, "PATH": VS_ROOT + os.pathsep + os.environ.get("PATH", "")},
        )
        if result.returncode == 0:
            version_line = result.stdout.strip().split("\n")[0]
            log(f"vspipe: {version_line}")
            return True
        else:
            stderr = result.stderr.strip()
            if "VSScript" in stderr or "Python" in stderr:
                log_error(
                    "VSPipe failed to initialize — VapourSynth portable may need "
                    "its bundled Python. Check that the portable zip was fully extracted."
                )
            else:
                log_error(f"vspipe --version failed: {stderr}")
            return False
    except FileNotFoundError:
        log_error("VSPipe.exe not executable")
        return False
    except subprocess.TimeoutExpired:
        log_error("vspipe --version timed out")
        return False


def verify_plugins() -> bool:
    """
    Test that plugins load by running a minimal VapourSynth script.
    This is optional — it may fail if VS portable's Python is not set up,
    but the plugins can still work with vspipe.
    """
    vspipe = os.path.join(VS_ROOT, "VSPipe.exe")
    if not os.path.isfile(vspipe):
        return False

    test_script = os.path.join(VS_ROOT, "_test_plugins.vpy")
    try:
        with open(test_script, "w") as f:
            f.write(
                'import vapoursynth as vs\n'
                'core = vs.core\n'
                'plugins = [p.namespace for p in core.plugins()]\n'
                'import sys\n'
                'print("Loaded plugins: " + ", ".join(sorted(plugins)), file=sys.stderr)\n'
                '# Check for key plugins\n'
                'has_trt = "trt" in plugins\n'
                'has_lsmas = "lsmas" in plugins\n'
                'print(f"  vstrt (TensorRT): {"YES" if has_trt else "NO"}", file=sys.stderr)\n'
                'print(f"  lsmas (L-SMASH):  {"YES" if has_lsmas else "NO"}", file=sys.stderr)\n'
                '# Must set an output for vspipe\n'
                'clip = core.std.BlankClip()\n'
                'clip.set_output()\n'
            )

        result = subprocess.run(
            [vspipe, test_script, "--info", "-"],
            capture_output=True, text=True, timeout=30,
            env={**os.environ, "PATH": VS_ROOT + os.pathsep + os.environ.get("PATH", "")},
        )

        if result.returncode == 0:
            for line in result.stderr.strip().split("\n"):
                if line.strip():
                    log(line.strip())
            return True
        else:
            log(f"Plugin test returned code {result.returncode}")
            if result.stderr.strip():
                for line in result.stderr.strip().split("\n")[:5]:
                    log(f"  {line}")
            return False

    except Exception as e:
        log(f"Plugin verification skipped: {e}")
        return False
    finally:
        if os.path.exists(test_script):
            os.remove(test_script)


# ---------------------------------------------------------------------------
# Engine build
# ---------------------------------------------------------------------------

def build_engine(clip_path: str = None) -> bool:
    """
    Build TensorRT engine by running encode.vpy on a test clip.
    This triggers vs-mlrt's engine build (first-run only).
    """
    vspipe = os.path.join(VS_ROOT, "VSPipe.exe")
    encode_vpy = os.path.join(PROJECT_ROOT, "remaster", "encode.vpy")

    if not os.path.isfile(vspipe):
        log_error("VSPipe.exe not found — install VapourSynth first")
        return False

    if not os.path.isfile(encode_vpy):
        log_error(f"encode.vpy not found at {encode_vpy}")
        return False

    # Find a test clip
    if clip_path is None:
        candidates = [
            os.path.join(PROJECT_ROOT, "data", "clip_mid_1080p.mp4"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                clip_path = c
                break

    if clip_path is None or not os.path.isfile(clip_path):
        log_error(
            "No test clip found for engine build. Provide one with --clip or "
            "place a clip at data/clip_mid_1080p.mp4"
        )
        return False

    # Check for ONNX model
    model_candidates = [
        os.path.join(PROJECT_ROOT, "checkpoints", "nafnet_w32_mid4",
                     "nafnet_w32mid4_1088x1920.onnx"),
        os.path.join(PROJECT_ROOT, "checkpoints", "nafnet_distill",
                     "nafnet_w64_1088x1920.onnx"),
    ]
    model_found = any(os.path.isfile(m) for m in model_candidates)
    if not model_found:
        log_error("No ONNX model found. Export one first (see cloud/modal_export_onnx_w32.py)")
        return False

    log(f"Building TensorRT engine (this takes 2-5 minutes)...")
    log(f"  Script: {encode_vpy}")
    log(f"  Clip:   {clip_path}")

    import time
    start = time.time()

    env = {**os.environ, "PATH": VS_ROOT + os.pathsep + os.environ.get("PATH", "")}

    # Run vspipe with encode.vpy, process just 1 frame to trigger engine build
    # The -e 0 flag processes only frame 0
    cmd = [
        vspipe, encode_vpy,
        "-a", f"input={clip_path}",
        "-e", "0",      # only process frame 0
        "-p",            # show progress
        "-c", "y4m",     # output format
        os.devnull,      # discard output (NUL on Windows, but os.devnull handles it)
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True, text=True, timeout=600,  # 10 min max
            env=env,
        )
    except subprocess.TimeoutExpired:
        log_error("Engine build timed out after 10 minutes")
        return False

    elapsed = time.time() - start

    if result.returncode == 0:
        log(f"Engine built successfully in {elapsed:.0f}s")

        # Report engine location
        for m in model_candidates:
            engine_dir = os.path.join(os.path.dirname(m), "engines")
            if os.path.isdir(engine_dir):
                engines = [f for f in os.listdir(engine_dir) if f.endswith(".engine")]
                if engines:
                    log(f"  Engine dir: {engine_dir}")
                    for e in engines:
                        size_mb = os.path.getsize(os.path.join(engine_dir, e)) / 1024 / 1024
                        log(f"  Engine: {e} ({size_mb:.0f} MB)")
        return True
    else:
        log_error(f"Engine build failed (exit code {result.returncode})")
        if result.stderr.strip():
            for line in result.stderr.strip().split("\n")[-10:]:
                log_error(f"  {line}")
        return False


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results: dict[str, bool]) -> None:
    """Print a summary of what was installed."""
    print()
    print("=" * 60)
    print("  Installation Summary")
    print("=" * 60)
    for component, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {component:30s} [{status}]")
    print("=" * 60)

    all_ok = all(results.values())
    if all_ok:
        print()
        print("  All components installed successfully.")
        print()
        print("  To encode a video:")
        print(f"    python remaster/encode.py input.mkv output.mkv")
        print()
        print("  VapourSynth root: " + VS_ROOT)
        vspipe = os.path.join(VS_ROOT, "VSPipe.exe")
        print("  vspipe:           " + vspipe)
    else:
        print()
        print("  Some components failed. Check the errors above.")
        print("  You can re-run this script to retry failed steps.")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Set up VapourSynth + vs-mlrt + source filters for video remastering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--build-engine", action="store_true",
        help="Build TensorRT engine after installation (takes 2-5 min)",
    )
    parser.add_argument(
        "--clip", default=None,
        help="Path to test clip for engine build (default: data/clip_mid_1080p.mp4)",
    )
    parser.add_argument(
        "--skip-prereqs", action="store_true",
        help="Skip prerequisite checks",
    )
    args = parser.parse_args()

    print()
    print("VapourSynth Setup")
    print("=" * 60)
    print(f"  Project root:  {PROJECT_ROOT}")
    print(f"  Install dir:   {VS_ROOT}")
    print(f"  Downloads dir: {DOWNLOADS_DIR}")
    print()

    # Prerequisites
    if not args.skip_prereqs:
        log("Checking prerequisites...")
        if not check_prerequisites():
            log_error("Prerequisites not met. Fix the issues above and retry.")
            log_error("Or pass --skip-prereqs to skip these checks.")
            sys.exit(1)
        print()

    results = {}

    # Step 1: VapourSynth
    log("--- VapourSynth Portable ---")
    results["VapourSynth " + VS_VERSION] = install_vapoursynth()
    print()

    # Step 2: vs-mlrt
    log("--- vs-mlrt (TensorRT backend) ---")
    results["vs-mlrt " + VSMLRT_VERSION] = install_vsmlrt()
    print()

    # Step 3: L-SMASH-Works
    log("--- L-SMASH-Works (source filter) ---")
    results["L-SMASH-Works"] = install_lsmash()
    print()

    # Step 4: Verify
    log("--- Verification ---")
    results["vspipe runs"] = verify_vspipe()
    verify_plugins()  # informational, don't gate on this
    print()

    # Optional: build engine
    if args.build_engine:
        log("--- TensorRT Engine Build ---")
        results["TensorRT engine"] = build_engine(args.clip)
        print()

    print_summary(results)

    # Exit code: 0 if all OK, 1 if any failed
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
