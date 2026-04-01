"""
Modal SCUNet Denoiser — cloud GPU pipeline for full episodes.

Upload a video, denoise all frames with batched SCUNet on a cloud GPU,
encode to high-quality 10-bit H.265, download the result.

Usage:
    modal run cloud/modal_denoise.py --input "C:/path/to/episode.mkv"
    modal run cloud/modal_denoise.py --input "C:/path/to/episode.mkv" --batch-size 8
    modal run cloud/modal_denoise.py --input "C:/path/to/episode.mkv" --crf 20 --model scunet_color_real_gan

GPU recommendations (for 1080p SCUNet fp16):
    T4  (16GB, ~$0.59/hr) — budget option, batch_size=3-4
    L4  (24GB, ~$0.80/hr) — best value, batch_size=6-8  [DEFAULT]
    A10G(24GB, ~$1.10/hr) — faster, batch_size=6-8
    A100(40GB, ~$2.10/hr) — overkill for this model, batch_size=12+

Local test (same pipeline, your GPU):
    python pipelines/denoise_batch.py --input episode.mkv --batch-size 1 --encoder hevc_nvenc --max-frames 100
"""
import modal
import os
import time

vol = modal.Volume.from_name("upscale-data", create_if_missing=True)
VOL_MOUNT = "/mnt/data"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "git", "libgl1", "libglib2.0-0",
        # ffmpeg build deps
        "build-essential", "yasm", "nasm", "pkg-config",
        "libx265-dev", "libnuma-dev",
    )
    # Build ffmpeg with NVENC + libx265 (NVENC only needs nv-codec-headers + driver)
    .run_commands(
        # nv-codec-headers: maps ffmpeg to NVIDIA driver's NVENC/NVDEC
        "git clone --depth 1 https://github.com/FFmpeg/nv-codec-headers.git /tmp/nvcodec"
        " && cd /tmp/nvcodec && make install && rm -rf /tmp/nvcodec",
    )
    .run_commands(
        # Build ffmpeg from GitHub mirror (git.ffmpeg.org is unreliable)
        "git clone --depth 1 --branch n7.1 https://github.com/FFmpeg/FFmpeg.git /tmp/ffmpeg-src"
        " && cd /tmp/ffmpeg-src"
        " && ./configure"
        "   --enable-nonfree --enable-gpl"
        "   --enable-nvenc --enable-libx265"
        "   --disable-doc --disable-debug --disable-static --enable-shared"
        " && make -j$(nproc)"
        " && make install"
        " && ldconfig"
        " && cd / && rm -rf /tmp/ffmpeg-src",
        "ffmpeg -encoders 2>/dev/null | grep nvenc | head -5",
    )
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "opencv-python-headless",
        "numpy",
        "einops",
        "timm",
    )
    .add_local_dir(
        "reference-code/SCUNet",
        remote_path="/root/SCUNet",
        ignore=["__pycache__/**", "*.pyc", "testsets/**", "results/**"],
    )
    .add_local_file("pipelines/denoise_batch.py", remote_path="/root/project/denoise_batch.py")
    .add_local_dir("lib", remote_path="/root/project/lib", ignore=["__pycache__/**", "*.pyc"])
)

app = modal.App("denoise-scunet", image=image)


@app.function(
    gpu="L4",
    volumes={VOL_MOUNT: vol},
    timeout=14400,  # 4 hours max
    memory=16384,   # 16GB RAM for ffmpeg encode buffer
)
def denoise_remote(
    input_path: str,
    output_path: str,
    model_name: str = "scunet_color_real_psnr",
    batch_size: int = 6,
    crf: int = 18,
    max_frames: int = -1,
):
    """Run denoise_batch pipeline on a cloud GPU."""
    import sys
    sys.path.insert(0, "/root/project")
    from denoise_batch import denoise_video

    # Ensure volume is synced with latest uploads
    vol.reload()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Verify input is accessible
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input not found on volume: {input_path}")
    print(f"Input file size: {os.path.getsize(input_path) / 1024**2:.1f} MB")

    result = denoise_video(
        input_path=input_path,
        output_path=output_path,
        model_name=model_name,
        batch_size=batch_size,
        crf=crf,
        encoder="hevc_nvenc",  # L4 NVENC hardware encoder
        max_frames=max_frames,
        use_compile=False,
        use_sdpa=False,
    )

    vol.commit()
    return result


@app.function(volumes={VOL_MOUNT: vol}, timeout=3600)
def mux_streams(denoised_path: str, original_path: str, final_path: str):
    """Mux denoised video with audio/subtitle streams from the original."""
    import subprocess
    vol.reload()
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    cmd = [
        "ffmpeg", "-hide_banner", "-y",
        "-i", denoised_path,
        "-i", original_path,
        "-map", "0:v:0",
        "-map", "1:a?",
        "-map", "1:s?",
        "-c:v", "copy",
        "-c:a", "copy",
        "-c:s", "copy",
        "-movflags", "+faststart",
        final_path,
    ]
    print(f"Muxing: {final_path}")
    subprocess.run(cmd, check=True)
    vol.commit()
    print("Mux complete.")


@app.local_entrypoint()
def main(
    input: str,
    output: str = "",
    model: str = "scunet_color_real_psnr",
    batch_size: int = 6,
    crf: int = 18,
    max_frames: int = -1,
):
    """
    Denoise a local video file using SCUNet on a Modal cloud GPU.

    Examples:
        modal run cloud/modal_denoise.py --input episode.mkv
        modal run cloud/modal_denoise.py --input episode.mkv --batch-size 8
        modal run cloud/modal_denoise.py --input episode.mkv --crf 20
    """
    import pathlib

    input_path = str(pathlib.Path(input).resolve())
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")

    if not output:
        p = pathlib.Path(input_path)
        output = str(p.with_stem(p.stem + "_denoised").with_suffix(".mkv"))

    input_size = os.path.getsize(input_path) / 1024**2
    print(f"Input:  {input_path} ({input_size:.0f} MB)")
    print(f"Output: {output}")
    print(f"batch_size: {batch_size}, CRF: {crf}, model: {model}")

    # Volume paths (relative to volume root for upload/download API)
    basename = os.path.basename(input_path)
    stem = pathlib.Path(input_path).stem
    vol_rel_input = f"/input/{basename}"
    vol_rel_output_video = f"/output/{stem}_denoised_video.mkv"
    vol_rel_output_final = f"/output/{stem}_denoised.mkv"
    # Container paths (volume root + mount point)
    ctr_input = f"{VOL_MOUNT}{vol_rel_input}"
    ctr_output_video = f"{VOL_MOUNT}{vol_rel_output_video}"
    ctr_output_final = f"{VOL_MOUNT}{vol_rel_output_final}"

    # Upload
    print(f"\nUploading {input_size:.0f} MB to Modal volume...")
    t0 = time.time()
    with vol.batch_upload(force=True) as batch:
        batch.put_file(input_path, vol_rel_input)
    print(f"  Upload done in {time.time() - t0:.0f}s")

    # Denoise on cloud GPU
    print(f"\nStarting denoise on L4...")
    result = denoise_remote.remote(
        input_path=ctr_input,
        output_path=ctr_output_video,
        model_name=model,
        batch_size=batch_size,
        crf=crf,
        max_frames=max_frames,
    )
    print(f"\nDenoise complete: {result['frames']} frames, "
          f"{result['elapsed_min']:.1f} min, {result['fps']:.1f} fps")

    # Mux audio/subs
    print("\nMuxing audio and subtitles from original...")
    mux_streams.remote(ctr_output_video, ctr_input, ctr_output_final)

    # Download (read_file uses volume-relative paths)
    print(f"\nDownloading result...")
    t0 = time.time()
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "wb") as f:
        for chunk in vol.read_file(vol_rel_output_final):
            f.write(chunk)
    dl_time = time.time() - t0
    out_size = os.path.getsize(output) / 1024**2
    print(f"  Downloaded {out_size:.0f} MB in {dl_time:.0f}s")

    print(f"\n{'=' * 60}")
    print(f"DONE: {output}")
    print(f"{'=' * 60}")
