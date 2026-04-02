"""
Generate training pairs using SCUNet GAN + detail transfer on Modal.

Target = SCUNet_GAN(frame) + alpha * high_pass(frame)

This produces sharper targets than SCUNet PSNR while preserving real
texture detail from the original (no hallucination from the high-pass).

Usage:
    modal run cloud/modal_generate_pairs.py --input-dir data/train_pairs/input --output-dir data/train_pairs_gan
    modal run cloud/modal_generate_pairs.py --input-dir data/val_pairs/input --output-dir data/val_pairs_gan
"""
import modal
import os
import time
import glob

vol = modal.Volume.from_name("upscale-data", create_if_missing=True)
VOL_MOUNT = "/mnt/data"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.7.1",
        "torchvision==0.22.1",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install("opencv-python-headless", "numpy", "einops", "timm")
    .add_local_dir(
        "reference-code/SCUNet",
        remote_path="/root/SCUNet",
        ignore=["__pycache__/**", "*.pyc", "testsets/**", "results/**"],
    )
)

app = modal.App("generate-gan-pairs", image=image)


@app.function(
    gpu="A10G",
    volumes={VOL_MOUNT: vol},
    timeout=7200,
    memory=16384,
)
def generate_pairs_remote(
    input_vol_dir: str,
    output_vol_dir: str,
    model_name: str = "scunet_color_real_gan",
    detail_alpha: float = 0.15,
    blur_sigma: float = 3.0,
):
    """Run SCUNet GAN + detail transfer on all input frames."""
    import sys
    import cv2
    import numpy as np
    import torch

    sys.path.insert(0, "/root/SCUNet")
    from models.network_scunet import SCUNet as net

    vol.reload()

    # Verify input
    input_files = sorted(glob.glob(os.path.join(input_vol_dir, "*.png")))
    if not input_files:
        raise FileNotFoundError(f"No PNG files in {input_vol_dir}")
    print(f"Input: {len(input_files)} frames from {input_vol_dir}")

    # Create output dirs
    out_input_dir = os.path.join(output_vol_dir, "input")
    out_target_dir = os.path.join(output_vol_dir, "target")
    os.makedirs(out_input_dir, exist_ok=True)
    os.makedirs(out_target_dir, exist_ok=True)

    # Skip already processed
    existing = set(os.path.basename(f) for f in glob.glob(os.path.join(out_target_dir, "*.png")))
    remaining = [(i, f) for i, f in enumerate(input_files) if os.path.basename(f) not in existing]
    print(f"Already processed: {len(existing)}, remaining: {len(remaining)}")

    if not remaining:
        print("All frames already processed!")
        vol.commit()
        return {"frames": len(input_files), "generated": 0}

    # Load SCUNet GAN
    device = "cuda"
    model_path = f"/root/SCUNet/model_zoo/{model_name}.pth"
    print(f"Loading {model_name}...")
    model = net(in_nc=3, config=[4, 4, 4, 4, 4, 4, 4], dim=64)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval().half().to(device)
    for p in model.parameters():
        p.requires_grad = False
    print(f"  VRAM: {torch.cuda.memory_allocated() / 1024**2:.0f}MB")

    start = time.time()
    for idx, (i, inp_path) in enumerate(remaining):
        fname = os.path.basename(inp_path)

        # Read input
        frame_bgr = cv2.imread(inp_path, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            print(f"  ERROR: Could not read {inp_path}")
            continue
        frame_f32 = frame_bgr.astype(np.float32) / 255.0

        # Run SCUNet GAN
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inp_t = torch.from_numpy(frame_rgb.transpose(2, 0, 1).copy()).float().unsqueeze(0).half().to(device) / 255.0
        with torch.no_grad():
            out_t = model(inp_t).clamp(0, 1)
        gan_rgb = (out_t.squeeze(0).float().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        gan_bgr = cv2.cvtColor(gan_rgb, cv2.COLOR_RGB2BGR)
        gan_f32 = gan_bgr.astype(np.float32) / 255.0

        # Detail transfer: high_pass from original, blend into GAN output
        blurred = cv2.GaussianBlur(frame_f32, (0, 0), blur_sigma)
        high_freq = frame_f32 - blurred
        target_f32 = np.clip(gan_f32 + detail_alpha * high_freq, 0, 1)
        target_bgr = (target_f32 * 255).astype(np.uint8)

        # Save pair
        cv2.imwrite(os.path.join(out_input_dir, fname), frame_bgr)
        cv2.imwrite(os.path.join(out_target_dir, fname), target_bgr)

        del inp_t, out_t
        if (idx + 1) % 20 == 0 or idx == 0:
            elapsed = time.time() - start
            fps = (idx + 1) / elapsed
            eta = (len(remaining) - idx - 1) / max(fps, 0.01)
            print(f"  [{idx+1}/{len(remaining)}] {fname} | "
                  f"{fps:.1f} fps | ETA: {eta / 60:.1f}min")

    vol.commit()
    total = time.time() - start
    print(f"\nDone: {len(remaining)} frames in {total / 60:.1f} min")
    return {"frames": len(input_files), "generated": len(remaining)}


@app.local_entrypoint()
def main(
    input_dir: str = "data/train_pairs/input",
    output_dir: str = "data/train_pairs_gan",
    detail_alpha: float = 0.15,
):
    """Upload input frames, generate GAN+detail targets on Modal."""
    import sys
    import subprocess
    import pathlib

    input_dir = os.path.abspath(input_dir)
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    input_files = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    print(f"Input: {len(input_files)} frames from {input_dir}")

    # Determine volume paths from output_dir name
    output_name = os.path.basename(os.path.abspath(output_dir))
    vol_input_dir = f"{VOL_MOUNT}/{output_name}_src"
    vol_output_dir = f"{VOL_MOUNT}/{output_name}"

    # Upload input frames
    print(f"\nUploading {len(input_files)} frames...")
    t0 = time.time()
    with vol.batch_upload(force=True) as batch:
        for f in input_files:
            fname = os.path.basename(f)
            batch.put_file(f, f"/{output_name}_src/{fname}")
    print(f"  Upload done in {time.time() - t0:.0f}s")

    # Generate pairs
    print(f"\nGenerating GAN+detail pairs (alpha={detail_alpha})...")
    result = generate_pairs_remote.remote(
        input_vol_dir=vol_input_dir,
        output_vol_dir=vol_output_dir,
        detail_alpha=detail_alpha,
    )
    print(f"Generated {result['generated']} pairs")

    # Download results using modal volume get (parallel, doesn't hang)
    import subprocess
    local_output = os.path.abspath(output_dir)
    os.makedirs(os.path.join(local_output, "input"), exist_ok=True)
    os.makedirs(os.path.join(local_output, "target"), exist_ok=True)

    python = sys.executable
    for subdir in ["input", "target"]:
        local_subdir = os.path.join(local_output, subdir)
        print(f"\nDownloading {subdir}...")
        t0 = time.time()
        subprocess.run(
            [python, "-m", "modal", "volume", "get", "--force",
             "upscale-data", f"{output_name}/{subdir}", local_subdir],
            check=True, env={**os.environ, "PYTHONUTF8": "1"},
        )
        n = len(glob.glob(os.path.join(local_subdir, "*.png")))
        print(f"  Downloaded {n} files in {time.time() - t0:.0f}s")

    n_input = len(glob.glob(os.path.join(local_output, "input", "*.png")))
    n_target = len(glob.glob(os.path.join(local_output, "target", "*.png")))
    print(f"\n{'='*60}")
    print(f"DONE: {local_output}")
    print(f"  input/  — {n_input} frames")
    print(f"  target/ — {n_target} frames")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
