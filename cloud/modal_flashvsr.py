"""
Run FlashVSR on Modal cloud GPU.
Separate app from the main runner to avoid conflicts with other agents.

Usage:
    # Upload a test clip first
    modal volume put flashvsr-data ./data/clip_mid_1080p.mp4 /inputs/clip_mid_1080p.mp4

    # Run FlashVSR
    modal run modal_flashvsr.py --input /inputs/clip_mid_1080p.mp4

    # Download result
    modal volume get flashvsr-data /outputs/ ./data/flashvsr_results/
"""
import modal

vol = modal.Volume.from_name("flashvsr-data", create_if_missing=True)
VOL_MOUNT = "/mnt/data"

# Step 1: Base image with ALL system deps including build tools
base = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install(
        "ffmpeg", "git", "libgl1", "libglib2.0-0",
        "ninja-build", "build-essential", "g++",  # needed for CUDA kernel compilation
    )
    .pip_install("packaging", "ninja", "wheel", "setuptools")
    .pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0",
        "torchaudio==2.6.0",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "einops", "huggingface-hub", "numpy==1.26.4",
        "opencv-python-headless", "pillow", "safetensors",
        "sentencepiece", "transformers==4.46.2",
        "imageio", "imageio-ffmpeg", "tqdm", "accelerate", "peft", "modelscope",
    )
)

# Step 2: Clone repos (cached, no GPU needed)
with_repos = (
    base
    .run_commands(
        "git clone https://github.com/OpenImagingLab/FlashVSR /root/FlashVSR",
        "git clone https://github.com/mit-han-lab/Block-Sparse-Attention /root/Block-Sparse-Attention",
    )
    .env({
        "PYTHONPATH": "/root/FlashVSR:/root/FlashVSR/examples/WanVSR",
        "CXX": "g++",
        "CC": "gcc",
    })
)

# Step 3: Compile CUDA kernels (no GPU needed — cross-compiles for sm_80 using nvcc)
# MAX_JOBS=8 for parallel compilation of 25 kernels
with_bsa = (
    with_repos
    .run_commands(
        "cd /root/Block-Sparse-Attention && "
        "BLOCK_SPARSE_ATTN_CUDA_ARCHS='80' MAX_JOBS=8 python setup.py install",
    )
)

# Step 4: All FlashVSR runtime deps (from their requirements.txt + discovered)
with_deps = with_bsa.pip_install(
    "modelscope", "ftfy", "protobuf==3.20.3", "pandas",
    "datasets", "pytorch-lightning", "torchmetrics", "torchsde",
)

# Step 5: Download model weights
image = (
    with_deps
    .run_commands(
        "pip install huggingface_hub[cli]",
        "mkdir -p /root/FlashVSR/examples/WanVSR/FlashVSR-v1.1",
        "huggingface-cli download JunhaoZhuang/FlashVSR-v1.1 --local-dir /root/FlashVSR/examples/WanVSR/FlashVSR-v1.1",
    )
)

app = modal.App("flashvsr-runner", image=image)


@app.function(
    gpu="A100-80GB",
    volumes={VOL_MOUNT: vol},
    timeout=3600,
)
def run_flashvsr(input_path: str, output_name: str = "flashvsr_output.mp4", scale: float = 4.0):
    """Run FlashVSR v1.1 tiny on a video clip."""
    import os, sys, time
    import torch
    import numpy as np
    from PIL import Image

    sys.path.insert(0, "/root/FlashVSR")
    sys.path.insert(0, "/root/FlashVSR/examples/WanVSR")
    os.chdir("/root/FlashVSR/examples/WanVSR")

    from diffsynth import ModelManager, FlashVSRTinyPipeline
    from utils.utils import Buffer_LQ4x_Proj
    from utils.TCDecoder import build_tcdecoder

    # Resolve input path
    full_input = os.path.join(VOL_MOUNT, input_path.lstrip("/")) if not input_path.startswith(VOL_MOUNT) else input_path
    if not os.path.exists(full_input):
        raise FileNotFoundError(f"Input not found: {full_input}")

    output_dir = os.path.join(VOL_MOUNT, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    full_output = os.path.join(output_dir, output_name)

    print(f"Input: {full_input}")
    print(f"Output: {full_output}")
    print(f"Scale: {scale}x")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

    # Load pipeline
    print("\nLoading FlashVSR v1.1 tiny pipeline...")
    t0 = time.time()
    mm = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    mm.load_models([
        "./FlashVSR-v1.1/diffusion_pytorch_model_streaming_dmd.safetensors",
    ])
    pipe = FlashVSRTinyPipeline.from_model_manager(mm, device="cuda")

    pipe.denoising_model().LQ_proj_in = Buffer_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to("cuda", dtype=torch.bfloat16)
    lq_path = "./FlashVSR-v1.1/LQ_proj_in.ckpt"
    if os.path.exists(lq_path):
        pipe.denoising_model().LQ_proj_in.load_state_dict(torch.load(lq_path, map_location="cpu"), strict=True)
    pipe.denoising_model().LQ_proj_in.to("cuda")

    multi_scale_channels = [512, 256, 128, 128]
    pipe.TCDecoder = build_tcdecoder(new_channels=multi_scale_channels, new_latent_channels=16+768)
    pipe.TCDecoder.load_state_dict(torch.load("./FlashVSR-v1.1/TCDecoder.ckpt"), strict=False)

    pipe.to("cuda")
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.init_cross_kv()
    pipe.load_models_to_device(["dit", "vae"])
    print(f"Pipeline loaded in {time.time()-t0:.1f}s")

    # Import helpers (filename has dots so can't be imported normally)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "flashvsr_helpers",
        "/root/FlashVSR/examples/WanVSR/infer_flashvsr_v1.1_tiny.py"
    )
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    prepare_input_tensor = helpers.prepare_input_tensor
    tensor2video = helpers.tensor2video
    save_video = helpers.save_video

    # Prepare input — limit frames to avoid OOM
    MAX_FRAMES = 49  # 8n+1 format: 49 = 8*6+1, gives 45 output frames (~2 sec)
    print(f"\nPreparing input (max {MAX_FRAMES} frames)...")
    dtype, device = torch.bfloat16, "cuda"

    # Monkey-patch to limit frame count
    original_prepare = prepare_input_tensor
    def limited_prepare(path, scale=4, dtype=torch.bfloat16, device='cuda'):
        import imageio
        if os.path.isfile(path) and path.lower().endswith(('.mp4','.mov','.avi','.mkv')):
            rdr = imageio.get_reader(path)
            first = Image.fromarray(rdr.get_data(0)).convert('RGB')
            w0, h0 = first.size
            meta = {}
            try: meta = rdr.get_meta_data()
            except: pass
            fps_val = meta.get('fps', 24)
            fps = int(round(fps_val)) if isinstance(fps_val, (int, float)) else 24

            sW, sH, tW, tH = helpers.compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=128)
            print(f"  Original: {w0}x{h0}, Scaled: {sW}x{sH}, Target: {tW}x{tH}")

            frames = []
            for i in range(min(MAX_FRAMES, rdr.count_frames())):
                img = Image.fromarray(rdr.get_data(i)).convert('RGB')
                img_out = helpers.upscale_then_center_crop(img, scale=scale, tW=tW, tH=tH)
                frames.append(helpers.pil_to_tensor_neg1_1(img_out, dtype, device))
            rdr.close()

            # Pad to 8n+1
            while len(frames) % 8 != 1:
                frames.append(frames[-1])

            F = len(frames)
            vid = torch.stack(frames, 0).permute(1,0,2,3).unsqueeze(0)
            print(f"  Loaded {F} frames ({F-4} output)")
            return vid, tH, tW, F, fps
        return original_prepare(path, scale, dtype, device)

    LQ, th, tw, F, fps = limited_prepare(full_input, scale=scale, dtype=dtype, device=device)
    print(f"  Input tensor: {LQ.shape}, target: {tw}x{th}, {F} frames, {fps}fps")

    # Run inference
    print("\nRunning FlashVSR...")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    video = pipe(
        prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=0,
        LQ_video=LQ, num_frames=F, height=th, width=tw,
        is_full_block=False, if_buffer=True,
        topk_ratio=2.0 * 768 * 1280 / (th * tw),
        kv_ratio=3.0,
        local_range=11,
        color_fix=True,
    )

    elapsed = time.time() - t0
    peak_vram = torch.cuda.max_memory_allocated() / 1024**3
    actual_frames = F - 4  # last 4 are padding
    print(f"\nInference done: {actual_frames} frames in {elapsed:.1f}s ({actual_frames/elapsed:.2f} fps)")
    print(f"Peak VRAM: {peak_vram:.1f}GB")

    # Save output
    print("\nSaving video...")
    video_frames = tensor2video(video)
    save_video(video_frames[:actual_frames], full_output, fps=fps, quality=6)
    print(f"Saved: {full_output}")

    vol.commit()
    return {
        "output": full_output,
        "frames": actual_frames,
        "fps": actual_frames / elapsed,
        "peak_vram_gb": peak_vram,
        "elapsed_s": elapsed,
    }


@app.local_entrypoint()
def main(input: str = "/inputs/clip_mid_1080p.mp4", output: str = "flashvsr_output.mp4", scale: float = 2.0):
    """
    Usage:
        modal run modal_flashvsr.py --input /inputs/clip_mid_1080p.mp4 --scale 2
        modal run modal_flashvsr.py --input /inputs/clip_480p.mp4 --scale 4
    """
    print(f"Dispatching FlashVSR to A100-80GB...")
    print(f"  Input: {input}")
    print(f"  Scale: {scale}x")
    result = run_flashvsr.remote(input, output, scale)
    print(f"\nResult: {result}")
    print(f"\nDownload with: modal volume get flashvsr-data /outputs/{output} .")
