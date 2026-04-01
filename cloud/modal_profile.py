"""
NAFNet speed profiling on Modal — measure fps/memory/quality for optimization experiments.

Runs a quick benchmark (~50 frames) on a Modal GPU and reports results in TSV format
for the speed optimization experiment log (bench/speed-opt/results.tsv).

Usage:
    PYTHONUTF8=1 C:/Users/sean/miniconda3/envs/upscale/python.exe -m modal run cloud/modal_profile.py

    # Override defaults:
    ... cloud/modal_profile.py --batch-size 4 --compile --compile-mode max-autotune --gpu A100
    ... cloud/modal_profile.py --channels-last --cudnn-benchmark
    ... cloud/modal_profile.py --tensorrt

See bench/speed-opt/EXPERIMENT.md for the full experiment guide.
"""
import modal
import os
import time

vol = modal.Volume.from_name("upscale-data", create_if_missing=True)
VOL_MOUNT = "/mnt/data"

# Base image with PyTorch + OpenCV (no ffmpeg build needed for profiling)
base_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1", "libglib2.0-0", "ffmpeg")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install("opencv-python-headless", "numpy")
    .add_local_dir("lib", remote_path="/root/project/lib", ignore=["__pycache__/**", "*.pyc"])
)

# TensorRT image (heavier, only used when --tensorrt flag is set)
trt_image = (
    base_image
    .pip_install("torch-tensorrt==2.5.0", extra_index_url="https://download.pytorch.org/whl/cu121")
)

app = modal.App("nafnet-profile", image=base_image)


@app.function(
    gpu="L4",
    volumes={VOL_MOUNT: vol},
    timeout=1800,
    memory=8192,
)
def profile_l4(
    checkpoint_path: str,
    clip_path: str,
    num_frames: int = 50,
    batch_size: int = 1,
    use_compile: bool = False,
    compile_mode: str = "reduce-overhead",
    use_channels_last: bool = False,
    use_cudnn_benchmark: bool = False,
    use_tensorrt: bool = False,
):
    return _run_profile(
        checkpoint_path=checkpoint_path,
        clip_path=clip_path,
        num_frames=num_frames,
        batch_size=batch_size,
        use_compile=use_compile,
        compile_mode=compile_mode,
        use_channels_last=use_channels_last,
        use_cudnn_benchmark=use_cudnn_benchmark,
        use_tensorrt=use_tensorrt,
        gpu_name="L4",
        gpu_hourly_rate=0.80,
    )


@app.function(
    gpu="A10G",
    volumes={VOL_MOUNT: vol},
    timeout=1800,
    memory=8192,
)
def profile_a10g(
    checkpoint_path: str,
    clip_path: str,
    num_frames: int = 50,
    batch_size: int = 1,
    use_compile: bool = False,
    compile_mode: str = "reduce-overhead",
    use_channels_last: bool = False,
    use_cudnn_benchmark: bool = False,
    use_tensorrt: bool = False,
):
    return _run_profile(
        checkpoint_path=checkpoint_path,
        clip_path=clip_path,
        num_frames=num_frames,
        batch_size=batch_size,
        use_compile=use_compile,
        compile_mode=compile_mode,
        use_channels_last=use_channels_last,
        use_cudnn_benchmark=use_cudnn_benchmark,
        use_tensorrt=use_tensorrt,
        gpu_name="A10G",
        gpu_hourly_rate=1.10,
    )


@app.function(
    gpu="A100",
    volumes={VOL_MOUNT: vol},
    timeout=1800,
    memory=8192,
)
def profile_a100(
    checkpoint_path: str,
    clip_path: str,
    num_frames: int = 50,
    batch_size: int = 1,
    use_compile: bool = False,
    compile_mode: str = "reduce-overhead",
    use_channels_last: bool = False,
    use_cudnn_benchmark: bool = False,
    use_tensorrt: bool = False,
):
    return _run_profile(
        checkpoint_path=checkpoint_path,
        clip_path=clip_path,
        num_frames=num_frames,
        batch_size=batch_size,
        use_compile=use_compile,
        compile_mode=compile_mode,
        use_channels_last=use_channels_last,
        use_cudnn_benchmark=use_cudnn_benchmark,
        use_tensorrt=use_tensorrt,
        gpu_name="A100",
        gpu_hourly_rate=2.78,
    )


def _run_profile(
    checkpoint_path: str,
    clip_path: str,
    num_frames: int,
    batch_size: int,
    use_compile: bool,
    compile_mode: str,
    use_channels_last: bool,
    use_cudnn_benchmark: bool,
    use_tensorrt: bool,
    gpu_name: str,
    gpu_hourly_rate: float,
):
    """Core profiling logic — runs inside the Modal container."""
    import sys
    sys.path.insert(0, "/root/project")

    import subprocess
    import numpy as np
    import torch
    from lib.nafnet_arch import NAFNet

    vol.reload()

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not os.path.exists(clip_path):
        raise FileNotFoundError(f"Clip not found: {clip_path}")

    # --- Environment info ---
    gpu_props = torch.cuda.get_device_properties(0)
    print(f"GPU: {gpu_props.name} ({gpu_props.total_memory / 1024**3:.1f} GB)")
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
    print(f"cuDNN: {torch.backends.cudnn.version()}")
    print(f"Config: bs={batch_size}, compile={use_compile}({compile_mode}), "
          f"trt={use_tensorrt}, channels_last={use_channels_last}, cudnn_bench={use_cudnn_benchmark}")

    # --- Config ---
    if use_cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
        print("cudnn.benchmark = True")

    # --- Load model ---
    model = NAFNet(
        img_channel=3, width=64,
        middle_blk_num=12,
        enc_blk_nums=[2, 2, 4, 8],
        dec_blk_nums=[2, 2, 2, 2],
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = ckpt.get("params", ckpt.get("params_ema", ckpt.get("model", ckpt.get("state_dict", ckpt))))
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model = model.cuda().half()

    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)
        print("channels_last memory format enabled")

    model_mb = torch.cuda.memory_allocated() / 1024**2
    print(f"Model loaded: {model_mb:.0f} MB VRAM")

    # --- Compile or TensorRT ---
    if use_tensorrt:
        try:
            import torch_tensorrt
            # Use torch_tensorrt.compile for simplicity
            example_input = torch.randn(batch_size, 3, 1088, 1920, device="cuda", dtype=torch.float16)
            if use_channels_last:
                example_input = example_input.to(memory_format=torch.channels_last)
            print(f"Building TensorRT engine (batch_size={batch_size}, fp16, 1920x1088)...")
            t0 = time.time()
            model = torch_tensorrt.compile(
                model,
                inputs=[example_input],
                enabled_precisions={torch.float16},
                truncate_long_and_double=True,
            )
            print(f"  TensorRT build done in {time.time() - t0:.1f}s")
        except Exception as e:
            import traceback
            print(f"  TensorRT FAILED: {e}")
            traceback.print_exc()
            print("  Falling back to eager")
            use_tensorrt = False
    elif use_compile:
        print(f"torch.compile mode={compile_mode}, warming up at 1920x1088...")
        model = torch.compile(model, mode=compile_mode)
        # Warmup at actual resolution (critical — different shapes = different compilations)
        dummy = torch.randn(batch_size, 3, 1088, 1920, device="cuda", dtype=torch.float16)
        if use_channels_last:
            dummy = dummy.to(memory_format=torch.channels_last)
        t0 = time.time()
        with torch.no_grad():
            _ = model(dummy)
        torch.cuda.synchronize()
        warmup_time = time.time() - t0
        print(f"  Compile warmup: {warmup_time:.1f}s")
        # Second warmup pass (CUDA graphs may need 2 passes)
        with torch.no_grad():
            _ = model(dummy)
        torch.cuda.synchronize()
        del dummy
        torch.cuda.empty_cache()

    # --- Extract frames from clip ---
    print(f"\nExtracting {num_frames} frames from {clip_path}...")
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", clip_path],
        capture_output=True, text=True,
    )
    w, h = [int(x) for x in probe.stdout.strip().split(",")]
    print(f"  Video: {w}x{h}")

    frame_bytes = w * h * 3
    read_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", clip_path,
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-vframes", str(num_frames),
        "pipe:1",
    ]
    proc = subprocess.Popen(read_cmd, stdout=subprocess.PIPE, bufsize=frame_bytes * 4)

    frames = []
    while len(frames) < num_frames:
        raw = proc.stdout.read(frame_bytes)
        if len(raw) < frame_bytes:
            break
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
        frames.append(frame)
    proc.stdout.close()
    proc.wait()
    print(f"  Extracted {len(frames)} frames")

    if not frames:
        raise RuntimeError("No frames extracted")

    # Save first input frame for PSNR comparison
    first_input = frames[0].copy()

    # --- Reset memory stats ---
    torch.cuda.reset_peak_memory_stats()

    # --- Batched inference ---
    print(f"\nProfiling: batch_size={batch_size}, compile={use_compile}, "
          f"trt={use_tensorrt}, channels_last={use_channels_last}")

    warmup_batches = 3  # skip first N batches for timing
    all_times = []
    first_output = None
    processed = 0
    batch_idx = 0

    try:
        while processed < len(frames):
            batch_end = min(processed + batch_size, len(frames))
            batch_frames = frames[processed:batch_end]

            # Pad incomplete final batch
            while len(batch_frames) < batch_size:
                batch_frames.append(batch_frames[-1])

            batch_np = np.stack(batch_frames)
            batch_t = torch.from_numpy(batch_np.transpose(0, 3, 1, 2).copy()).half() / 255.0
            batch_t = batch_t.cuda()
            if use_channels_last:
                batch_t = batch_t.to(memory_format=torch.channels_last)

            torch.cuda.synchronize()
            t0 = time.perf_counter()

            with torch.no_grad():
                out_t = model(batch_t)

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            if batch_idx >= warmup_batches:
                # Only count real frames (not padding)
                real_count = batch_end - processed
                all_times.append((elapsed, real_count))

            # Progress log every batch
            fps_so_far = sum(n for _, n in all_times) / sum(t for t, _ in all_times) if all_times else 0
            peak_so_far = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  batch {batch_idx}: {len(batch_frames)} frames in {elapsed:.3f}s "
                  f"({len(batch_frames)/elapsed:.1f} fps this batch, "
                  f"{fps_so_far:.1f} fps avg, peak {peak_so_far:.1f} GB)")

            # Save first output for PSNR
            if first_output is None:
                first_output = (out_t[0].clamp(0, 1) * 255).byte().cpu().numpy().transpose(1, 2, 0)

            del batch_t, out_t
            processed = batch_end
            batch_idx += 1

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n  OOM at batch_size={batch_size}!")
            peak_gb = torch.cuda.max_memory_allocated() / 1024**3
            return {
                "fps": 0.0,
                "peak_gb": peak_gb,
                "psnr_db": 0.0,
                "cost_ep": 999.0,
                "gpu": gpu_name,
                "batch_size": batch_size,
                "status": "crash",
                "description": f"OOM at bs={batch_size} (peak {peak_gb:.1f} GB)",
            }
        raise

    # --- Compute metrics ---
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3

    total_time = sum(t for t, _ in all_times)
    total_frames = sum(n for _, n in all_times)
    fps = total_frames / total_time if total_time > 0 else 0

    # PSNR: output vs input (measures how much the model changed the image)
    psnr_db = 0.0
    if first_output is not None:
        diff = first_input.astype(np.float64) - first_output.astype(np.float64)
        mse = np.mean(diff ** 2)
        if mse > 0:
            psnr_db = 10 * np.log10(255.0 ** 2 / mse)
        else:
            psnr_db = float("inf")

    # Cost estimate: 61K frames for a Firefly episode
    cost_ep = 61000 / fps / 3600 * gpu_hourly_rate if fps > 0 else 999.0

    # --- Build description ---
    parts = []
    if use_tensorrt:
        parts.append("TensorRT fp16")
    elif use_compile:
        parts.append(f"compile({compile_mode})")
    else:
        parts.append("eager")
    parts.append(f"bs={batch_size}")
    parts.append(gpu_name)
    if use_channels_last:
        parts.append("channels_last")
    if use_cudnn_benchmark:
        parts.append("cudnn.bench")
    description = " + ".join(parts)

    result = {
        "fps": round(fps, 2),
        "peak_gb": round(peak_gb, 1),
        "psnr_db": round(psnr_db, 2),
        "cost_ep": round(cost_ep, 2),
        "gpu": gpu_name,
        "batch_size": batch_size,
        "status": "keep" if fps > 0 else "crash",
        "description": description,
    }

    # --- Print results ---
    print(f"\n{'=' * 70}")
    print(f"  RESULTS: {description}")
    print(f"{'=' * 70}")
    print(f"  FPS:         {fps:.2f}")
    print(f"  Peak VRAM:   {peak_gb:.1f} GB")
    print(f"  PSNR:        {psnr_db:.2f} dB")
    print(f"  Cost/episode: ${cost_ep:.2f}")
    print(f"  Frames timed: {total_frames} (skipped {warmup_batches} warmup batches)")
    print(f"{'=' * 70}")
    print(f"\nTSV line (copy to bench/speed-opt/results.tsv):")
    print(f"COMMIT\t{fps:.2f}\t{peak_gb:.1f}\t{psnr_db:.2f}\t${cost_ep:.2f}\t{gpu_name}\t{batch_size}\t{result['status']}\t{description}")

    return result


@app.local_entrypoint()
def main(
    batch_size: int = 1,
    num_frames: int = 50,
    compile: bool = False,
    compile_mode: str = "reduce-overhead",
    channels_last: bool = False,
    cudnn_benchmark: bool = False,
    tensorrt: bool = False,
    gpu: str = "L4",
):
    """
    Profile NAFNet inference speed on Modal.

    Examples:
        # Baseline
        modal run cloud/modal_profile.py

        # With channels_last + cudnn benchmark
        modal run cloud/modal_profile.py --channels-last --cudnn-benchmark

        # With torch.compile
        modal run cloud/modal_profile.py --compile --channels-last --cudnn-benchmark

        # With torch.compile max-autotune
        modal run cloud/modal_profile.py --compile --compile-mode max-autotune --channels-last

        # With TensorRT (requires trt_image)
        modal run cloud/modal_profile.py --tensorrt --channels-last

        # Different GPU
        modal run cloud/modal_profile.py --gpu A100 --batch-size 4 --compile

        # Sweep batch sizes
        modal run cloud/modal_profile.py --batch-size 4 --compile --channels-last
    """
    import pathlib

    # Checkpoint and clip paths on Modal volume
    ckpt_local = "checkpoints/nafnet_distill/nafnet_best.pth"
    clip_local = "data/clip_mid_1080p.mp4"

    # Resolve local paths
    ckpt_local = str(pathlib.Path(ckpt_local).resolve())
    clip_local = str(pathlib.Path(clip_local).resolve())

    if not os.path.exists(ckpt_local):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_local}")
    if not os.path.exists(clip_local):
        raise FileNotFoundError(f"Clip not found: {clip_local}")

    # Volume paths
    vol_ckpt = "/checkpoints/nafnet_best.pth"
    vol_clip = "/input/clip_mid_1080p.mp4"
    ctr_ckpt = f"{VOL_MOUNT}{vol_ckpt}"
    ctr_clip = f"{VOL_MOUNT}{vol_clip}"

    # Upload
    print("Uploading checkpoint and clip to Modal volume...")
    t0 = time.time()
    with vol.batch_upload(force=True) as batch:
        batch.put_file(ckpt_local, vol_ckpt)
        batch.put_file(clip_local, vol_clip)
    print(f"  Upload done in {time.time() - t0:.0f}s")

    # Select GPU function
    kwargs = dict(
        checkpoint_path=ctr_ckpt,
        clip_path=ctr_clip,
        num_frames=num_frames,
        batch_size=batch_size,
        use_compile=compile,
        compile_mode=compile_mode,
        use_channels_last=channels_last,
        use_cudnn_benchmark=cudnn_benchmark,
        use_tensorrt=tensorrt,
    )

    print(f"\nStarting profile on {gpu}...")
    print(f"  batch_size={batch_size}, compile={compile}, compile_mode={compile_mode}")
    print(f"  channels_last={channels_last}, cudnn_benchmark={cudnn_benchmark}, tensorrt={tensorrt}")

    gpu_funcs = {"L4": profile_l4, "A10G": profile_a10g, "A100": profile_a100}
    fn = gpu_funcs.get(gpu.upper())
    if fn is None:
        raise ValueError(f"Unsupported GPU: {gpu}. Choose from: {list(gpu_funcs.keys())}")

    result = fn.remote(**kwargs)

    print(f"\nDone. Result: {result}")
