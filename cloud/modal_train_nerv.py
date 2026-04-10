"""
Modal training wrapper for NeRV video denoising.

NeRV fits a neural network to a video clip — the network IS the video representation.
The spectral bias of the network denoises by learning structure before noise.

Usage:
    # Phase 1a: Reproduce exp37 on T4 (sanity check)
    modal run cloud/modal_train_nerv.py --gpu T4 --frames 16 --batch-size 1 \
        --run-name modal-repro-16f-bs1

    # Phase 1c: Scale to 64 frames
    modal run cloud/modal_train_nerv.py --gpu T4 --frames 64 --batch-size 4 \
        --epochs 200 --run-name modal-64f-bs4

    # Phase 2: Larger model on L4
    modal run cloud/modal_train_nerv.py --gpu L4 --frames 128 --batch-size 4 \
        --fc-dim 170 --enc-dim 24 --weight-decay 0.02 --epochs 300 \
        --run-name modal-170fc-128f

    # Resume previous run (automatically finds latest.pth on volume)
    modal run cloud/modal_train_nerv.py --skip-upload --resume \
        --run-name modal-repro-16f-bs1

    # Skip upload after first run
    modal run cloud/modal_train_nerv.py --skip-upload --gpu T4 ...
"""
import modal
import os
import glob
import time

vol = modal.Volume.from_name("upscale-data", create_if_missing=True)
VOL_MOUNT = "/mnt/data"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.11.0",
        "torchvision",
        extra_index_url="https://download.pytorch.org/whl/cu130",
    )
    .pip_install(
        "numpy",
        "Pillow",
        "matplotlib",
        "prodigyopt",
        "wandb",
        "pytorch_msssim",
    )
    .add_local_file(
        "tools/train_nerv.py",
        remote_path="/root/project/tools/train_nerv.py",
    )
)

app = modal.App("nerv-denoise", image=image)


@app.function(
    gpu="T4",
    volumes={VOL_MOUNT: vol},
    timeout=7200,  # 2 hours max
    memory=16384,
    secrets=[modal.Secret.from_name("wandb-api-key")],
)
def train_nerv(
    # Data
    data_remote_dir: str = "/nerv-data/micro_gop_01",
    num_frames: int = 16,
    # Model
    fc_dim: int = 120,
    enc_dim: int = 16,
    enc_strides: str = "3,3,2,2,2",
    dec_strides: str = "3,3,2,2,2",
    dec_blks: str = "1,1,1,1,1",
    enc_blocks: int = 1,
    # Skip connections
    skip_connections: bool = True,
    skip_scale_init: float = 0.1,
    skip_dropout: float = 0.0,
    # Training
    epochs: int = 150,
    batch_size: int = 1,
    optimizer: str = "prodigy",
    lr: float = 2e-4,
    weight_decay: float = 0.01,
    d_coef: float = 1.0,
    # Loss
    loss: str = "l1_freq",
    pixel_weight: float = 10.0,
    edge_weight: float = 0.5,
    asym_edge_weight: float = 0.5,
    frame2frame: bool = False,
    # Logging
    wandb_enabled: bool = True,
    wandb_project: str = "remaster",
    wandb_entity: str = "seantempesta",
    run_name: str = "",
    # Resume
    resume: bool = False,
    fresh_optimizer: bool = False,
    # Misc
    max_time: int = 3600,
    print_interval: int = 10,
    checkpoint_dir: str = "",
):
    """Train NeRV on Modal GPU."""
    import sys
    import os as _os
    _os.environ["PYTHONUNBUFFERED"] = "1"  # flush prints immediately for log tailing
    sys.path.insert(0, "/root/project")
    sys.path.insert(0, "/root/project/tools")

    # Reload volume to see latest data and checkpoints from previous runs
    vol.reload()

    # Set up data path
    data_dir = os.path.join(VOL_MOUNT, data_remote_dir.lstrip("/"))
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data dir not found: {data_dir}. Upload frames first.")

    frames = sorted(glob.glob(os.path.join(data_dir, "*.png")))
    print(f"Found {len(frames)} frames in {data_dir}")
    if num_frames > 0 and len(frames) > num_frames:
        print(f"Using first {num_frames} frames")

    # Set up output/checkpoint dir ON the volume
    if not checkpoint_dir:
        checkpoint_dir = os.path.join(VOL_MOUNT, "nerv-output", run_name or "default")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Handle resume: construct path to latest.pth on the volume
    resume_path = None
    if resume:
        resume_path = os.path.join(checkpoint_dir, "latest.pth")
        if os.path.exists(resume_path):
            print(f"Resuming from {resume_path}")
        else:
            print(f"  No checkpoint found at {resume_path}, starting fresh")
            resume_path = None

    # Build args namespace to match train_nerv.py's parse_args
    import argparse
    args = argparse.Namespace(
        data_dir=data_dir,
        num_frames=num_frames,
        fc_dim=fc_dim,
        enc_dim=enc_dim,
        enc_strides=enc_strides,
        dec_strides=dec_strides,
        dec_blks=dec_blks,
        enc_blocks=enc_blocks,
        reduce=1.2,
        lower_width=12,
        skip_connections=skip_connections,
        skip_scale_init=skip_scale_init,
        skip_stages=None,
        skip_dropout=skip_dropout,
        epochs=epochs,
        batch_size=batch_size,
        optimizer=optimizer,
        lr=lr,
        weight_decay=weight_decay,
        d_coef=d_coef,
        loss=loss,
        pixel_weight=pixel_weight,
        edge_weight=edge_weight,
        asym_edge_weight=asym_edge_weight,
        residual_flatness_weight=0.0,
        its_alpha=0.0,
        frame2frame=frame2frame,
        its_threshold=35.0,
        grad_checkpoint=False,
        late_layer_decay=0.0,
        max_time=max_time,
        output_dir=checkpoint_dir,
        ckpt_interval=10,
        print_interval=print_interval,
        resume=resume_path,
        fresh_optimizer=fresh_optimizer,
        wandb=wandb_enabled and os.environ.get("WANDB_API_KEY", "") != "",
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_run_name=run_name,
        run_name=run_name,
        snapshot_freq=30,
        model_size=0,
        cond_ch=32,
        # Phase 2 args (kept for compat, ramp is now default)
        phase2_epoch=0,
        phase2_pixel_weight=1.0,
        phase2_edge_weight=2.0,
        phase2_asym_edge_weight=2.0,
        device="cuda",
    )

    # Import and run training
    from train_nerv import train
    train(args)

    # Commit volume so checkpoints persist between runs
    vol.commit()
    print("Training complete, volume committed.")

    # Print results summary
    metrics_path = os.path.join(checkpoint_dir, "metrics.jsonl")
    if os.path.exists(metrics_path):
        import json
        with open(metrics_path) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        if lines:
            best = max(lines, key=lambda x: x.get("val_psnr", 0))
            last = lines[-1]
            print(f"\n{'='*60}")
            print(f"RESULTS: {run_name}")
            print(f"  Best val_psnr: {best['val_psnr']:.2f} dB (epoch {best['epoch']})")
            print(f"  Final val_psnr: {last['val_psnr']:.2f} dB")
            print(f"  Final sharpness_ratio: {last.get('val_sharpness_ratio', 'N/A')}")
            print(f"  Final residual_structure: {last.get('val_residual_structure', 'N/A')}")
            print(f"  Epochs: {last['epoch'] + 1}")
            print(f"  Params: {last.get('params_M', 'N/A')}M")
            print(f"{'='*60}")

    # Return checkpoint dir so local_entrypoint can download
    return checkpoint_dir


@app.local_entrypoint()
def main(
    # Data
    data_dir: str = "E:/upscale-data/nerv-test/micro_gop_01",
    remote_dir: str = "/nerv-data/micro_gop_01",
    skip_upload: bool = False,
    # GPU
    gpu: str = "T4",
    # Model
    fc_dim: int = 120,
    enc_dim: int = 16,
    enc_strides: str = "3,3,2,2,2",
    dec_strides: str = "3,3,2,2,2",
    dec_blks: str = "1,1,1,1,1",
    enc_blocks: int = 1,
    # Skip connections
    skip_connections: bool = True,
    skip_scale_init: float = 0.1,
    skip_dropout: float = 0.0,
    # Training
    frames: int = 16,
    epochs: int = 150,
    batch_size: int = 1,
    optimizer: str = "prodigy",
    lr: float = 2e-4,
    weight_decay: float = 0.01,
    d_coef: float = 1.0,
    # Loss
    loss: str = "l1_freq",
    pixel_weight: float = 10.0,
    edge_weight: float = 0.5,
    asym_edge_weight: float = 0.5,
    frame2frame: bool = False,
    # Resume
    resume: bool = False,
    fresh_optimizer: bool = False,
    # Logging
    wandb: bool = True,
    run_name: str = "",
    # Misc
    max_time: int = 3600,
):
    """Local entrypoint: upload data then launch training on Modal."""

    # Auto-generate run name if not specified
    if not run_name:
        run_name = f"modal-nerv-{frames}f-bs{batch_size}-fc{fc_dim}"

    # Checkpoint dir on volume (must match remote function's logic)
    vol_ckpt_dir = f"{VOL_MOUNT}/nerv-output/{run_name}"

    print(f"NeRV Modal Training: {run_name}")
    print(f"  GPU: {gpu}")
    print(f"  Frames: {frames}, Batch: {batch_size}, Epochs: {epochs}")
    print(f"  Model: fc_dim={fc_dim}, enc_dim={enc_dim}")
    print(f"  Loss: pixel_weight={pixel_weight}, edge={edge_weight}, asym_edge={asym_edge_weight}")
    print(f"  Optimizer: {optimizer}, wd={weight_decay}, d_coef={d_coef}")
    print(f"  Resume: {'YES (from latest.pth on volume)' if resume else 'NO (fresh start)'}")
    print(f"  Fresh optimizer: {fresh_optimizer}")
    print(f"  Checkpoint dir: {vol_ckpt_dir}")
    print(f"  Data: {data_dir} -> {remote_dir}")
    print(f"  Max time: {max_time}s ({max_time/60:.0f} min)")
    print()

    # Upload data
    # Normalize remote_dir: strip leading slashes (Git Bash converts /path to C:/path)
    remote_dir = remote_dir.replace("\\", "/").lstrip("/")
    if not remote_dir.startswith("nerv-data"):
        remote_dir = f"nerv-data/{remote_dir}"

    if not skip_upload:
        local_frames = sorted(glob.glob(os.path.join(data_dir, "*.png")))
        print(f"Uploading {len(local_frames)} frames from {data_dir}...")
        with vol.batch_upload(force=True) as batch:
            for f in local_frames:
                batch.put_file(os.path.abspath(f), f"/{remote_dir}/{os.path.basename(f)}")
        print(f"Upload complete: {len(local_frames)} frames -> /{remote_dir}")
    else:
        print("Skipping upload (--skip-upload)")

    # Launch training (GPU is set in @app.function decorator -- change there for different GPU)
    print(f"\nLaunching training on Modal (GPU set in decorator)...")
    result_path = train_nerv.remote(
        data_remote_dir=f"/{remote_dir}",
        num_frames=frames,
        fc_dim=fc_dim,
        enc_dim=enc_dim,
        enc_strides=enc_strides,
        dec_strides=dec_strides,
        dec_blks=dec_blks,
        enc_blocks=enc_blocks,
        skip_connections=skip_connections,
        skip_scale_init=skip_scale_init,
        skip_dropout=skip_dropout,
        epochs=epochs,
        batch_size=batch_size,
        optimizer=optimizer,
        lr=lr,
        weight_decay=weight_decay,
        d_coef=d_coef,
        loss=loss,
        pixel_weight=pixel_weight,
        edge_weight=edge_weight,
        asym_edge_weight=asym_edge_weight,
        frame2frame=frame2frame,
        resume=resume,
        fresh_optimizer=fresh_optimizer,
        wandb_enabled=wandb,
        run_name=run_name,
        max_time=max_time,
        checkpoint_dir=vol_ckpt_dir,
    )
    print(f"\nTraining complete. Checkpoints at: {result_path}")

    # Download checkpoints from volume to local
    ckpt_rel = vol_ckpt_dir.replace(VOL_MOUNT + "/", "")
    local_ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ckpt_rel)
    os.makedirs(local_ckpt_dir, exist_ok=True)

    for name in ["best_psnr.pth", "best_denoise.pth", "latest.pth", "metrics.jsonl"]:
        vol_path = f"/{ckpt_rel}/{name}"
        local_path = os.path.join(local_ckpt_dir, name)
        try:
            with open(local_path, "wb") as f:
                vol.read_file_into_fileobj(vol_path, f)
            size_mb = os.path.getsize(local_path) / 1024**2
            print(f"  Downloaded {name} ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"  Could not download {name}: {e}")

    # Download visualization images
    local_vis_dir = os.path.join(local_ckpt_dir, "vis")
    os.makedirs(local_vis_dir, exist_ok=True)
    try:
        vis_prefix = f"{ckpt_rel}/vis/"
        vis_count = 0
        for entry in vol.listdir(vis_prefix):
            name = entry.path.split("/")[-1]
            if not name.endswith(".png"):
                continue
            local_path = os.path.join(local_vis_dir, name)
            try:
                with open(local_path, "wb") as f:
                    vol.read_file_into_fileobj(f"/{vis_prefix}{name}", f)
                vis_count += 1
            except Exception:
                pass
        if vis_count > 0:
            print(f"  Downloaded {vis_count} visualization images to {local_vis_dir}")
    except Exception as e:
        print(f"  Could not list vis dir: {e}")

    print(f"\n{'='*60}")
    print(f"DONE. Checkpoints in: {local_ckpt_dir}")
    print(f"{'='*60}")
