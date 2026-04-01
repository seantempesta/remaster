"""
Modal Denoiser — cloud GPU pipeline for full episodes (SCUNet or NAFNet).

Upload a video, denoise all frames with batched SCUNet or distilled NAFNet
on a cloud GPU, encode to high-quality 10-bit H.265, download the result.

Usage:
    # SCUNet (default)
    modal run cloud/modal_denoise.py --input "C:/path/to/episode.mkv"
    modal run cloud/modal_denoise.py --input "C:/path/to/episode.mkv" --batch-size 8

    # NAFNet (distilled — faster)
    modal run cloud/modal_denoise.py --input "C:/path/to/episode.mkv" --model nafnet --checkpoint checkpoints/nafnet_distill/nafnet_best.pth
    modal run cloud/modal_denoise.py --input "C:/path/to/episode.mkv" --model nafnet --checkpoint checkpoints/nafnet_distill/nafnet_best.pth --compile

GPU recommendations (1080p fp16):
    T4  (16GB, ~$0.59/hr) — budget option, batch_size=3-4
    L4  (24GB, ~$0.80/hr) — best value, batch_size=6-8  [DEFAULT]
    A10G(24GB, ~$1.10/hr) — faster, batch_size=6-8
    A100(40GB, ~$2.10/hr) — overkill for this model, batch_size=12+

Local test (same pipeline, your GPU):
    python pipelines/denoise_batch.py --input episode.mkv --batch-size 1 --encoder hevc_nvenc --max-frames 100
    python pipelines/denoise_nafnet.py --input episode.mkv --checkpoint checkpoints/nafnet_distill/nafnet_best.pth --compile
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
    .add_local_file("pipelines/denoise_nafnet.py", remote_path="/root/project/denoise_nafnet.py")
    .add_local_dir("lib", remote_path="/root/project/lib", ignore=["__pycache__/**", "*.pyc"])
)

app = modal.App("denoise-video", image=image)


@app.function(
    gpu="L4",
    volumes={VOL_MOUNT: vol},
    timeout=14400,  # 4 hours max
    memory=16384,   # 16GB RAM for ffmpeg encode buffer
)
def denoise_remote(
    input_path: str,
    output_path: str,
    model_type: str = "scunet",
    model_name: str = "scunet_color_real_psnr",
    checkpoint_path: str = "",
    batch_size: int = 6,
    crf: int = 18,
    max_frames: int = -1,
    use_compile: bool = False,
):
    """Run denoise pipeline on a cloud GPU (SCUNet or NAFNet)."""
    import sys
    sys.path.insert(0, "/root/project")

    # Persist torch.compile / Triton caches to volume (avoids 5+ min warmup)
    cache_dir = f"{VOL_MOUNT}/torch_compile_cache"
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
    os.environ["TRITON_CACHE_DIR"] = f"{cache_dir}/triton"

    # Ensure volume is synced with latest uploads
    vol.reload()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Verify input is accessible
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input not found on volume: {input_path}")
    print(f"Input file size: {os.path.getsize(input_path) / 1024**2:.1f} MB")

    if model_type == "nafnet":
        from denoise_nafnet import denoise_video
        if not checkpoint_path:
            raise ValueError("--checkpoint is required for nafnet model")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"NAFNet checkpoint not found: {checkpoint_path}")
        result = denoise_video(
            input_path=input_path,
            output_path=output_path,
            checkpoint_path=checkpoint_path,
            batch_size=batch_size,
            crf=crf,
            encoder="hevc_nvenc",
            max_frames=max_frames,
            fp16=True,
            use_compile=use_compile,
        )
    else:
        from denoise_batch import denoise_video
        result = denoise_video(
            input_path=input_path,
            output_path=output_path,
            model_name=model_name,
            batch_size=batch_size,
            crf=crf,
            encoder="hevc_nvenc",
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
    model: str = "scunet",
    model_name: str = "scunet_color_real_psnr",
    checkpoint: str = "",
    batch_size: int = 6,
    crf: int = 18,
    max_frames: int = -1,
    compile: bool = False,
):
    """
    Denoise a local video file on a Modal cloud GPU.

    Examples:
        # SCUNet (default)
        modal run cloud/modal_denoise.py --input episode.mkv
        modal run cloud/modal_denoise.py --input episode.mkv --batch-size 8

        # NAFNet (distilled)
        modal run cloud/modal_denoise.py --input episode.mkv --model nafnet --checkpoint checkpoints/nafnet_distill/nafnet_best.pth
        modal run cloud/modal_denoise.py --input episode.mkv --model nafnet --checkpoint checkpoints/nafnet_distill/nafnet_best.pth --compile
    """
    import pathlib

    input_path = str(pathlib.Path(input).resolve())
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")

    if model == "nafnet" and not checkpoint:
        raise ValueError("--checkpoint is required when --model nafnet")
    if checkpoint:
        checkpoint = str(pathlib.Path(checkpoint).resolve())
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    # NAFNet uses more intermediate memory per frame (full-res convolutions)
    # than SCUNet (windowed attention), so default to smaller batches
    if batch_size == 6 and model == "nafnet":
        batch_size = 2

    if not output:
        p = pathlib.Path(input_path)
        suffix = "_nafnet" if model == "nafnet" else "_denoised"
        output = str(p.with_stem(p.stem + suffix).with_suffix(".mkv"))

    input_size = os.path.getsize(input_path) / 1024**2
    print(f"Input:  {input_path} ({input_size:.0f} MB)")
    print(f"Output: {output}")
    print(f"Model: {model}, batch_size: {batch_size}, CRF: {crf}, compile: {compile}")

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

    # Upload video (and checkpoint if NAFNet)
    print(f"\nUploading {input_size:.0f} MB to Modal volume...")
    t0 = time.time()
    ctr_checkpoint = ""
    with vol.batch_upload(force=True) as batch:
        batch.put_file(input_path, vol_rel_input)
        if model == "nafnet" and checkpoint:
            ckpt_basename = os.path.basename(checkpoint)
            vol_rel_ckpt = f"/checkpoints/{ckpt_basename}"
            batch.put_file(checkpoint, vol_rel_ckpt)
            ctr_checkpoint = f"{VOL_MOUNT}{vol_rel_ckpt}"
            print(f"  Uploading checkpoint: {ckpt_basename}")
    print(f"  Upload done in {time.time() - t0:.0f}s")

    # Denoise on cloud GPU
    print(f"\nStarting denoise on L4 ({model})...")
    result = denoise_remote.remote(
        input_path=ctr_input,
        output_path=ctr_output_video,
        model_type=model,
        model_name=model_name,
        checkpoint_path=ctr_checkpoint,
        batch_size=batch_size,
        crf=crf,
        max_frames=max_frames,
        use_compile=compile,
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
