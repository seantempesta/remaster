"""
Modal UAV evaluation -- test Upscale-A-Video as training target generator.

Downloads pretrained weights, runs inference on sequential frames from
data/nerv-test/clip_01/, generates side-by-side comparisons.

Usage:
    modal run cloud/modal_uav_eval.py
    modal run cloud/modal_uav_eval.py --skip-upload
    modal run cloud/modal_uav_eval.py --noise-level 80 --steps 20

    # Windows: prefix with PYTHONUTF8=1 if you get encoding errors
"""
import modal
import os
import time

vol = modal.Volume.from_name("upscale-data", create_if_missing=True)
VOL_MOUNT = "/mnt/data"
WEIGHTS_BASE = f"{VOL_MOUNT}/uav_weights"
WEIGHTS_DIR = f"{WEIGHTS_BASE}/upscale_a_video"
EVAL_DIR = f"{VOL_MOUNT}/uav_eval"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.0.1",
        "torchvision==0.15.2",
        index_url="https://download.pytorch.org/whl/cu118",
    )
    .pip_install(
        "diffusers==0.16.0",
        "transformers==4.28.1",
        "accelerate==0.18.0",
        "huggingface-hub==0.23.5",
        "einops>=0.6.1",
        "rotary-embedding-torch==0.2.3",
        "imageio==2.25.0",
        "imageio-ffmpeg==0.4.8",
        "opencv-python-headless",
        "numpy==1.24.3",
        "scipy",
        "tqdm",
        "pyfiglet",
        "sentencepiece==0.1.99",
        "gdown",
    )
    .add_local_dir(
        "reference-code/Upscale-A-Video/models_video",
        remote_path="/root/uav/models_video",
    )
    .add_local_dir(
        "reference-code/Upscale-A-Video/configs",
        remote_path="/root/uav/configs",
    )
    .add_local_file(
        "reference-code/Upscale-A-Video/utils.py",
        remote_path="/root/uav/utils.py",
    )
)

app = modal.App("remaster-uav-eval", image=image)


# ---------------------------------------------------------------------------
# Phase 1: Download weights (CPU only -- no GPU cost)
# ---------------------------------------------------------------------------
@app.function(
    volumes={VOL_MOUNT: vol},
    timeout=1800,
    memory=4096,
    cpu=2,
)
def setup_weights():
    """Download UAV pretrained weights to Modal volume."""
    vol.reload()

    unet_path = os.path.join(WEIGHTS_DIR, "unet", "unet_video.bin")
    if os.path.exists(unet_path):
        size_mb = os.path.getsize(unet_path) / 1024**2
        print(f"Weights already on volume (unet_video.bin = {size_mb:.0f} MB)")
        return True

    # Check if zip was already downloaded (from a previous failed run)
    import zipfile
    import shutil

    zip_candidates = []
    for root, dirs, files in os.walk(WEIGHTS_BASE):
        for f in files:
            if f.endswith(".zip"):
                zip_candidates.append(os.path.join(root, f))

    if zip_candidates:
        print(f"Found existing zip: {zip_candidates[0]}")
    else:
        print("Downloading UAV weights from Google Drive...")
        t0 = time.time()

        import gdown

        folder_url = "https://drive.google.com/drive/folders/1O8pbeR1hsRlFUU8O4EULe-lOKNGEWZl1"
        os.makedirs(WEIGHTS_BASE, exist_ok=True)

        try:
            gdown.download_folder(
                folder_url, output=WEIGHTS_BASE, quiet=False, use_cookies=False,
            )
        except Exception as e:
            print(f"ERROR: gdown download failed: {e}")
            return False

        elapsed = time.time() - t0
        print(f"Download finished in {elapsed:.0f}s")

        # Find the zip
        for root, dirs, files in os.walk(WEIGHTS_BASE):
            for f in files:
                if f.endswith(".zip"):
                    zip_candidates.append(os.path.join(root, f))

    if not zip_candidates:
        print("ERROR: No zip file found after download")
        return False

    # Extract the zip
    zip_path = zip_candidates[0]
    size_gb = os.path.getsize(zip_path) / 1024**3
    print(f"Extracting {zip_path} ({size_gb:.1f} GB)...")
    t0 = time.time()

    extract_dir = os.path.join(WEIGHTS_BASE, "_extract")
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    print(f"Extracted in {time.time() - t0:.0f}s")

    # Find where the actual model files are (unet/, vae/, etc.)
    # Walk extract dir to find unet_video.bin
    found_unet = None
    for root, dirs, files in os.walk(extract_dir):
        if "unet_video.bin" in files:
            # The parent of unet/ is our target
            found_unet = os.path.dirname(root)
            break

    if found_unet:
        print(f"Found model files at: {found_unet}")
        # Move to the expected location
        if os.path.exists(WEIGHTS_DIR):
            shutil.rmtree(WEIGHTS_DIR)
        shutil.move(found_unet, WEIGHTS_DIR)
    else:
        # Fallback: show what was extracted
        print("Could not find unet_video.bin. Extracted structure:")
        for root, dirs, files in os.walk(extract_dir):
            level = root.replace(extract_dir, "").count(os.sep)
            indent = "  " * level
            print(f"{indent}{os.path.basename(root)}/")
            if level < 4:
                for f in files[:10]:
                    print(f"{indent}  {f}")

    # Clean up zip and extract dir
    for zp in zip_candidates:
        if os.path.exists(zp):
            os.remove(zp)
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir, ignore_errors=True)
    # Clean up any nested gdown artifacts
    for item in os.listdir(WEIGHTS_BASE):
        full = os.path.join(WEIGHTS_BASE, item)
        if os.path.isdir(full) and item != "upscale_a_video":
            shutil.rmtree(full, ignore_errors=True)

    # Verify key files
    ok = True
    for name in ["unet/unet_video.bin", "vae/vae_3d.bin",
                  "scheduler/scheduler_config.json"]:
        path = os.path.join(WEIGHTS_DIR, name)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1024**2
            print(f"  OK  {name} ({size_mb:.1f} MB)")
        else:
            print(f"  MISSING  {name}")
            ok = False

    vol.commit()
    return ok


# ---------------------------------------------------------------------------
# Phase 2: Run inference (GPU)
# ---------------------------------------------------------------------------
@app.function(
    gpu="A100-80GB",
    volumes={VOL_MOUNT: vol},
    timeout=7200,
    memory=32768,
)
def run_eval(
    noise_level: int = 120,
    guidance_scale: int = 6,
    inference_steps: int = 30,
    num_frames: int = 60,
    prompt: str = "clean, sharp, high quality film",
    n_prompt: str = "blur, worst quality, compression artifacts",
    use_video_vae: bool = False,
    tile_size: int = 192,
    tile_overlap: int = 64,
    input_scale: float = 1.0,
):
    """Run UAV inference on frames uploaded to the volume."""
    import sys
    sys.path.insert(0, "/root/uav")

    import torch
    import torch.nn.functional as F
    import numpy as np
    import cv2
    from tqdm import tqdm
    from einops import rearrange

    vol.reload()

    input_dir = os.path.join(EVAL_DIR, "input")
    output_dir = os.path.join(EVAL_DIR, "output")
    compare_dir = os.path.join(EVAL_DIR, "compare")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(compare_dir, exist_ok=True)

    # ---- Verify inputs ----
    frame_files = sorted(f for f in os.listdir(input_dir) if f.endswith(".png"))
    frame_files = frame_files[:num_frames]
    if not frame_files:
        raise FileNotFoundError(f"No PNG frames in {input_dir}")
    print(f"Found {len(frame_files)} input frames")

    # ---- Verify weights ----
    unet_weights = os.path.join(WEIGHTS_DIR, "unet", "unet_video.bin")
    if not os.path.exists(unet_weights):
        raise FileNotFoundError(f"UNet weights not found at {unet_weights}. Run setup_weights first.")

    device = "cuda"

    # ---- Load models ----
    print("Loading models...")
    t0 = time.time()

    # Text encoder + tokenizer from HuggingFace (reliable, no Google Drive)
    from transformers import CLIPTextModel, CLIPTokenizer
    print("  text_encoder + tokenizer (from HuggingFace)...")
    text_encoder = CLIPTextModel.from_pretrained(
        "stabilityai/stable-diffusion-x4-upscaler",
        subfolder="text_encoder",
        torch_dtype=torch.float16,
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        "stabilityai/stable-diffusion-x4-upscaler",
        subfolder="tokenizer",
    )

    # Low-res scheduler from HuggingFace
    from diffusers import DDPMScheduler
    low_res_scheduler = DDPMScheduler.from_pretrained(
        "stabilityai/stable-diffusion-x4-upscaler",
        subfolder="low_res_scheduler",
    )

    # Custom UAV components
    from models_video.autoencoder_kl_cond_video import AutoencoderKLVideo
    from models_video.unet_video import UNetVideoModel
    from models_video.scheduling_ddim import DDIMScheduler
    from models_video.pipeline_upscale_a_video import VideoUpscalePipeline

    # VAE
    if use_video_vae:
        vae_cfg = "vae_video_config.json"
        vae_bin = "vae_video.bin"
    else:
        vae_cfg = "vae_3d_config.json"
        vae_bin = "vae_3d.bin"

    vae_config = os.path.join(WEIGHTS_DIR, "vae", vae_cfg)
    if not os.path.exists(vae_config):
        vae_config = f"/root/uav/configs/{vae_cfg}"
    vae_weights = os.path.join(WEIGHTS_DIR, "vae", vae_bin)
    print(f"  VAE ({vae_bin})...")
    vae = AutoencoderKLVideo.from_config(vae_config)
    vae.load_state_dict(torch.load(vae_weights, map_location="cpu"))

    # UNet
    unet_config = os.path.join(WEIGHTS_DIR, "unet", "unet_video_config.json")
    if not os.path.exists(unet_config):
        unet_config = "/root/uav/configs/unet_video_config.json"
    print("  UNet...")
    unet = UNetVideoModel.from_config(unet_config)
    unet.load_state_dict(torch.load(unet_weights, map_location="cpu"), strict=True)
    unet = unet.half()
    unet.eval()

    # Scheduler (custom DDIM with step_v0/step_vt)
    sched_config = os.path.join(WEIGHTS_DIR, "scheduler", "scheduler_config.json")
    if os.path.exists(sched_config):
        print("  Scheduler (from weights)...")
        scheduler = DDIMScheduler.from_config(sched_config)
    else:
        print("  Scheduler (default SD config)...")
        scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
            prediction_type="epsilon",
        )

    # Construct pipeline (no propagator for first test)
    pipeline = VideoUpscalePipeline(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        low_res_scheduler=low_res_scheduler,
        scheduler=scheduler,
        vae=vae,
        unet=unet,
        propagator=None,
    )
    pipeline = pipeline.to(device)

    # Monkey-patch VAE decode to process 1 frame at a time.
    # The VAE mid_block has spatial self-attention that is O(H*W)^2.
    # With T frames batched together (rearranged to batch dim), VRAM = T * (H*W)^2 * 4B.
    # At 256x144 with T=3: 3 * 36K^2 * 4B = 15GB -> OOM on A10G (24GB).
    # With T=1: 1 * 36K^2 * 4B = 5GB -> fits.
    _original_decode = pipeline.decode_latents_vsr

    def _decode_per_frame(latents, img, w_lr):
        results = []
        for t in range(latents.shape[2]):
            torch.cuda.empty_cache()
            r = _original_decode(latents[:, :, t:t+1], img[:, :, t:t+1], w_lr)
            results.append(r)
        return torch.cat(results, dim=2)

    pipeline.decode_latents_vsr = _decode_per_frame

    load_time = time.time() - t0
    print(f"Models loaded in {load_time:.1f}s")
    vram_mb = torch.cuda.memory_allocated() / 1024**2
    print(f"VRAM after model load: {vram_mb:.0f} MB")

    # ---- Load frames ----
    print(f"Loading {len(frame_files)} frames...")
    frames = []
    for fname in frame_files:
        img = cv2.imread(os.path.join(input_dir, fname))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)

    orig_h, orig_w = frames[0].shape[:2]
    print(f"  Original resolution: {orig_w}x{orig_h}")

    # Downscale input (must be divisible by 8 for UNet)
    target_w = int(orig_w * input_scale) // 8 * 8
    target_h = int(orig_h * input_scale) // 8 * 8
    if input_scale < 1.0:
        print(f"  Downscaling to {target_w}x{target_h} (scale={input_scale})")
    else:
        target_w = orig_w // 8 * 8
        target_h = orig_h // 8 * 8
        print(f"  Using full resolution: {target_w}x{target_h}")

    frames_scaled = []
    for img in frames:
        scaled = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        frames_scaled.append(scaled)

    # Convert to tensor: 1 C T H W, normalized to [-1, 1]
    vframes = np.stack(frames_scaled, axis=0)  # T H W C
    vframes = torch.from_numpy(vframes).permute(0, 3, 1, 2).float()  # T C H W
    vframes = (vframes / 255.0 - 0.5) * 2.0
    vframes = vframes.to(device)
    vframes = vframes.unsqueeze(0)
    vframes = rearrange(vframes, "b t c h w -> b c t h w").contiguous()

    b, c, t, h, w = vframes.shape
    output_h, output_w = h * 4, w * 4
    print(f"  Input tensor: {vframes.shape}")
    print(f"  Output will be: {output_w}x{output_h} (4x upscale)")

    # ---- Tiling setup ----
    import math
    tiles_x = math.ceil(w / tile_size)
    tiles_y = math.ceil(h / tile_size)

    # Check if last tile can be absorbed into overlap
    rm_end_pad_w, rm_end_pad_h = True, True
    if (tiles_x - 1) * tile_size + tile_overlap >= w:
        tiles_x -= 1
        rm_end_pad_w = False
    if (tiles_y - 1) * tile_size + tile_overlap >= h:
        tiles_y -= 1
        rm_end_pad_h = False

    total_tiles = tiles_x * tiles_y
    print(f"\nTiling: {tiles_x}x{tiles_y} = {total_tiles} tiles "
          f"(tile_size={tile_size}, overlap={tile_overlap})")

    # ---- Run tiled inference ----
    print(f"Running UAV inference:")
    print(f"  noise_level={noise_level}, guidance_scale={guidance_scale}, steps={inference_steps}")
    print(f"  prompt: '{prompt}'")

    output_tensor = vframes.new_zeros((b, c, t, output_h, output_w))
    generator = torch.Generator(device=device).manual_seed(42)

    torch.cuda.synchronize()
    t0 = time.time()

    for y in range(tiles_y):
        for x in range(tiles_x):
            tile_idx = y * tiles_x + x + 1
            print(f"  Tile [{tile_idx}/{total_tiles}] ({y+1},{x+1})...", end="", flush=True)
            torch.cuda.empty_cache()

            # Input tile coordinates
            ofs_x = x * tile_size
            ofs_y = y * tile_size
            input_start_x = ofs_x
            input_end_x = min(ofs_x + tile_size, w)
            input_start_y = ofs_y
            input_end_y = min(ofs_y + tile_size, h)

            # Add overlap padding (clipped to bounds)
            input_start_x_pad = max(input_start_x - tile_overlap, 0)
            input_end_x_pad = min(input_end_x + tile_overlap, w)
            input_start_y_pad = max(input_start_y - tile_overlap, 0)
            input_end_y_pad = min(input_end_y + tile_overlap, h)

            input_tile_width = input_end_x - input_start_x
            input_tile_height = input_end_y - input_start_y

            # Extract tile from all frames
            input_tile = vframes[:, :, :,
                                 input_start_y_pad:input_end_y_pad,
                                 input_start_x_pad:input_end_x_pad]

            # Run pipeline on tile
            tile_t0 = time.time()
            try:
                with torch.no_grad():
                    output_tile = pipeline(
                        prompt,
                        image=input_tile,
                        flows_bi=None,
                        generator=generator,
                        num_inference_steps=inference_steps,
                        guidance_scale=guidance_scale,
                        noise_level=noise_level,
                        negative_prompt=n_prompt,
                        propagation_steps=[],
                    ).images  # (1, C, T, H_out, W_out)
            except RuntimeError as e:
                vram_peak = torch.cuda.max_memory_allocated() / 1024**2
                print(f"\n  RUNTIME ERROR on tile {tile_idx}: {e}")
                print(f"  Tile input shape: {input_tile.shape}, VRAM peak: {vram_peak:.0f} MB")
                raise

            tile_time = time.time() - tile_t0

            # Output tile coordinates (4x input)
            output_start_x = input_start_x * 4
            output_start_y = input_start_y * 4
            if x == tiles_x - 1 and not rm_end_pad_w:
                output_end_x = output_w
            else:
                output_end_x = input_end_x * 4
            if y == tiles_y - 1 and not rm_end_pad_h:
                output_end_y = output_h
            else:
                output_end_y = input_end_y * 4

            # Crop overlap from output tile
            output_start_x_tile = (input_start_x - input_start_x_pad) * 4
            output_start_y_tile = (input_start_y - input_start_y_pad) * 4
            if x == tiles_x - 1 and not rm_end_pad_w:
                output_end_x_tile = output_start_x_tile + output_w - output_start_x
            else:
                output_end_x_tile = output_start_x_tile + input_tile_width * 4
            if y == tiles_y - 1 and not rm_end_pad_h:
                output_end_y_tile = output_start_y_tile + output_h - output_start_y
            else:
                output_end_y_tile = output_start_y_tile + input_tile_height * 4

            # Place tile into output
            output_tensor[:, :, :,
                          output_start_y:output_end_y,
                          output_start_x:output_end_x] = \
                output_tile[:, :, :,
                            output_start_y_tile:output_end_y_tile,
                            output_start_x_tile:output_end_x_tile]

            del output_tile, input_tile
            print(f" {tile_time:.1f}s")

    torch.cuda.synchronize()
    infer_time = time.time() - t0
    fps = len(frame_files) / infer_time
    print(f"\nInference done: {infer_time:.1f}s for {len(frame_files)} frames "
          f"({total_tiles} tiles, {fps:.2f} fps)")

    # Convert output: (1, C, T, H, W) -> numpy (T, H, W, C)
    output = output_tensor.squeeze(0)  # (C, T, H, W)
    output = rearrange(output, "c t h w -> t c h w")  # (T, C, H, W)
    output_np = (output / 2 + 0.5).clamp(0, 1)
    output_np = output_np.cpu().numpy()
    output_np = (output_np * 255).round().astype(np.uint8)
    output_np = np.transpose(output_np, (0, 2, 3, 1))  # (T, H, W, C)

    out_h, out_w = output_np.shape[1], output_np.shape[2]
    print(f"  Assembled output: {out_w}x{out_h}")

    # Downscale to original resolution for comparison
    if out_h != orig_h or out_w != orig_w:
        print(f"  Downscaling {out_w}x{out_h} -> {orig_w}x{orig_h}")
        resized = []
        for i in range(output_np.shape[0]):
            r = cv2.resize(output_np[i], (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
            resized.append(r)
        output_np = np.stack(resized, axis=0)

    # ---- Save output frames ----
    print(f"\nSaving {len(frame_files)} output frames...")
    for i, fname in enumerate(frame_files):
        out_path = os.path.join(output_dir, fname)
        out_bgr = cv2.cvtColor(output_np[i], cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, out_bgr)

    # ---- Generate side-by-side comparisons ----
    print("Generating comparisons...")
    # Pick every 10th frame for detailed comparison
    compare_indices = list(range(0, len(frame_files), 10))
    if compare_indices[-1] != len(frame_files) - 1:
        compare_indices.append(len(frame_files) - 1)

    for idx in compare_indices:
        orig = frames[idx]  # RGB
        enhanced = output_np[idx]  # RGB

        # Full side-by-side
        side_by_side = np.concatenate([orig, enhanced], axis=1)
        side_by_side_bgr = cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            os.path.join(compare_dir, f"compare_{idx:05d}_full.png"),
            side_by_side_bgr,
        )

        # Center crop detail comparison (480x480)
        crop_size = 480
        cy, cx = orig.shape[0] // 2, orig.shape[1] // 2
        y1, x1 = cy - crop_size // 2, cx - crop_size // 2
        orig_crop = orig[y1:y1 + crop_size, x1:x1 + crop_size]
        enh_crop = enhanced[y1:y1 + crop_size, x1:x1 + crop_size]
        detail = np.concatenate([orig_crop, enh_crop], axis=1)
        detail_bgr = cv2.cvtColor(detail, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            os.path.join(compare_dir, f"compare_{idx:05d}_detail.png"),
            detail_bgr,
        )

    # ---- Temporal consistency measurement ----
    print("Measuring temporal consistency...")
    # Compare consecutive frames in a static region (center 256x256)
    region_size = 256
    cy, cx = orig_h // 2, orig_w // 2
    ry1, rx1 = cy - region_size // 2, cx - region_size // 2

    diffs_original = []
    diffs_enhanced = []
    for i in range(len(frame_files) - 1):
        # Original frame-to-frame diff
        o1 = frames[i][ry1:ry1 + region_size, rx1:rx1 + region_size].astype(np.float32)
        o2 = frames[i + 1][ry1:ry1 + region_size, rx1:rx1 + region_size].astype(np.float32)
        diffs_original.append(np.mean(np.abs(o1 - o2)))

        # Enhanced frame-to-frame diff
        e1 = output_np[i][ry1:ry1 + region_size, rx1:rx1 + region_size].astype(np.float32)
        e2 = output_np[i + 1][ry1:ry1 + region_size, rx1:rx1 + region_size].astype(np.float32)
        diffs_enhanced.append(np.mean(np.abs(e1 - e2)))

    avg_diff_orig = np.mean(diffs_original)
    avg_diff_enh = np.mean(diffs_enhanced)
    print(f"  Avg frame-to-frame diff (center {region_size}x{region_size}):")
    print(f"    Original:  {avg_diff_orig:.2f}")
    print(f"    Enhanced:  {avg_diff_enh:.2f}")
    if avg_diff_enh < avg_diff_orig:
        print(f"    -> Enhanced is {(1 - avg_diff_enh/avg_diff_orig)*100:.1f}% more temporally consistent")
    else:
        print(f"    -> Enhanced is {(avg_diff_enh/avg_diff_orig - 1)*100:.1f}% LESS consistent (more flicker)")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"Results saved to volume at {EVAL_DIR}/")
    print(f"  output/   - {len(frame_files)} enhanced frames")
    print(f"  compare/  - {len(compare_indices)} side-by-side comparisons")
    print(f"  Speed: {fps:.2f} fps ({infer_time:.1f}s total)")
    print(f"{'='*60}")

    vol.commit()
    return {
        "num_frames": len(frame_files),
        "inference_time": infer_time,
        "fps": fps,
        "output_resolution": f"{output_np.shape[2]}x{output_np.shape[1]}",
        "temporal_diff_original": float(avg_diff_orig),
        "temporal_diff_enhanced": float(avg_diff_enh),
    }


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    noise_level: int = 120,
    guidance_scale: int = 6,
    steps: int = 30,
    num_frames: int = 8,
    skip_upload: bool = False,
    skip_weights: bool = False,
    use_video_vae: bool = False,
    prompt: str = "clean, sharp, high quality film",
    tile_size: int = 192,
    input_scale: float = 1.0,
):
    """
    Run UAV evaluation on Modal with tiling.

    Uploads frames from data/nerv-test/clip_02/, downloads weights if needed,
    runs tiled inference, downloads side-by-side comparisons.

    Examples:
        # 960x540 input, 15 tiles (default, ~$0.60)
        ... --input-scale 0.5

        # Full 1080p input, 60 tiles (~$2.30)
        ... --input-scale 1.0

        # Quick test, 4 frames
        ... --num-frames 4 --steps 10
    """
    import glob as glob_mod

    clip_dir = os.path.abspath("data/nerv-test/clip_02")
    if not os.path.isdir(clip_dir):
        raise FileNotFoundError(f"Clip directory not found: {clip_dir}")

    frame_files = sorted(glob_mod.glob(os.path.join(clip_dir, "*.png")))[:num_frames]
    print(f"UAV Evaluation (tiled)")
    print(f"  Clip: {clip_dir}")
    print(f"  Frames: {len(frame_files)}")
    print(f"  Input scale: {input_scale} -> ~{int(1920*input_scale)//8*8}x{int(1080*input_scale)//8*8}")
    print(f"  Tile size: {tile_size}")
    print(f"  Settings: noise={noise_level}, guidance={guidance_scale}, steps={steps}")

    # Upload frames to volume
    if not skip_upload:
        print(f"\nUploading {len(frame_files)} frames to Modal volume...")
        t0 = time.time()
        with vol.batch_upload(force=True) as batch:
            for f in frame_files:
                batch.put_file(f, f"/uav_eval/input/{os.path.basename(f)}")
        print(f"  Upload done in {time.time() - t0:.1f}s")
    else:
        print("\nSkipping frame upload (--skip-upload)")

    # Download weights if needed
    if not skip_weights:
        print("\nChecking/downloading model weights (CPU, no GPU cost)...")
        ok = setup_weights.remote()
        if not ok:
            print("ERROR: Weight download failed. Cannot continue.")
            return
        print("Weights ready.")
    else:
        print("\nSkipping weight check (--skip-weights)")

    # Run inference
    print(f"\nStarting inference on A100-80GB...")
    result = run_eval.remote(
        noise_level=noise_level,
        guidance_scale=guidance_scale,
        inference_steps=steps,
        num_frames=num_frames,
        prompt=prompt,
        use_video_vae=use_video_vae,
        tile_size=tile_size,
        input_scale=input_scale,
    )

    print(f"\nResults: {result}")

    # Download comparisons
    local_output = os.path.join("output", "uav_eval")
    os.makedirs(local_output, exist_ok=True)

    print(f"\nDownloading comparison images...")
    dl_count = 0
    try:
        for entry in vol.listdir("/uav_eval/compare/"):
            name = entry.path.split("/")[-1]
            if not name.endswith(".png"):
                continue
            local_path = os.path.join(local_output, name)
            with open(local_path, "wb") as f:
                vol.read_file_into_fileobj(f"/uav_eval/compare/{name}", f)
            dl_count += 1
    except Exception as e:
        print(f"  Error downloading: {e}")

    print(f"  Downloaded {dl_count} comparison images to {local_output}/")

    # Also download a few output frames for inspection
    print("Downloading sample output frames...")
    output_sample_dir = os.path.join(local_output, "frames")
    os.makedirs(output_sample_dir, exist_ok=True)
    sample_indices = [0, 10, 20, 30, 40, 50, 59]
    for idx in sample_indices:
        fname = f"{idx:05d}.png"
        try:
            local_path = os.path.join(output_sample_dir, fname)
            with open(local_path, "wb") as f:
                vol.read_file_into_fileobj(f"/uav_eval/output/{fname}", f)
            dl_count += 1
        except Exception:
            pass

    print(f"\n{'='*60}")
    print(f"DONE. Results in: {local_output}/")
    print(f"  compare/          - side-by-side (original | enhanced)")
    print(f"  frames/           - raw enhanced frames")
    print(f"{'='*60}")
