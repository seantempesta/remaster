"""
Modal training wrapper for NAFNet distillation.

Uploads training pairs to Modal volume, runs training on A10G,
downloads the best checkpoint.

Usage:
    # Upload data and train
    modal run cloud/modal_train.py --data-dir data/train_pairs --max-iters 50000

    # Resume from previous run
    modal run cloud/modal_train.py --data-dir data/train_pairs --max-iters 100000 --resume

    # Quick test
    modal run cloud/modal_train.py --data-dir data/train_pairs --max-iters 200 --val-freq 100
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
    .pip_install(
        "opencv-python-headless",
        "numpy",
    )
    .add_local_file("lib/nafnet_arch.py", remote_path="/root/project/lib/nafnet_arch.py")
    .add_local_file("lib/paths.py", remote_path="/root/project/lib/paths.py")
    .add_local_file("lib/__init__.py", remote_path="/root/project/lib/__init__.py")
    .add_local_file("training/train_nafnet.py", remote_path="/root/project/train_nafnet.py")
)

app = modal.App("train-nafnet-distill", image=image)


@app.function(
    gpu="H100",
    volumes={VOL_MOUNT: vol},
    timeout=28800,  # 8 hours max
    memory=65536,   # 64GB RAM
)
def train_remote(
    data_dir: str,
    checkpoint_dir: str,
    pretrained_path: str,
    max_iters: int = 50000,
    batch_size: int = 16,
    lr: float = 2e-4,
    loss: str = "charbonnier",
    resume: bool = False,
    val_freq: int = 1000,
    save_freq: int = 5000,
    crop_size: int = 384,
    grad_clip: float = 1.0,
    perceptual_weight: float = 0.0,
):
    """Run training on a cloud GPU."""
    import sys
    sys.path.insert(0, "/root/project")

    # Ensure volume is synced
    vol.reload()

    # Verify data exists
    input_dir = os.path.join(data_dir, "input")
    target_dir = os.path.join(data_dir, "target")
    n_input = len(glob.glob(os.path.join(input_dir, "*.png")))
    n_target = len(glob.glob(os.path.join(target_dir, "*.png")))
    print(f"Training data: {n_input} inputs, {n_target} targets")
    if n_input == 0 or n_target == 0:
        raise FileNotFoundError(f"No training pairs found in {data_dir}")

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Build args
    from train_nafnet import parse_args, train
    import argparse

    class Args:
        pass
    args = Args()
    args.data_dir = data_dir
    args.crop_size = crop_size
    args.batch_size = batch_size
    args.num_workers = 4
    args.pretrained = pretrained_path
    args.resume = os.path.join(checkpoint_dir, "nafnet_latest.pth") if resume else None
    args.max_iters = max_iters
    args.lr = lr
    args.eta_min = 1e-7
    args.weight_decay = 0.0
    args.warmup_iters = min(500, max_iters // 10)
    args.grad_clip = grad_clip
    args.loss = loss
    args.perceptual_weight = perceptual_weight
    args.amp = True
    args.checkpoint_dir = checkpoint_dir
    args.print_freq = 50
    args.val_freq = val_freq
    args.save_freq = save_freq
    args.device = "cuda"

    train(args)

    vol.commit()
    print("Training complete, volume committed.")

    # Return path to best model
    best_path = os.path.join(checkpoint_dir, "nafnet_best.pth")
    if os.path.exists(best_path):
        return best_path
    return os.path.join(checkpoint_dir, "nafnet_final.pth")


@app.local_entrypoint()
def main(
    data_dir: str = "data/train_pairs",
    max_iters: int = 50000,
    batch_size: int = 16,
    lr: float = 2e-4,
    loss: str = "charbonnier",
    resume: bool = False,
    val_freq: int = 1000,
    save_freq: int = 5000,
    crop_size: int = 384,
    grad_clip: float = 1.0,
    perceptual_weight: float = 0.0,
):
    """
    Upload training data and run NAFNet distillation training on Modal A10G.

    Examples:
        modal run cloud/modal_train.py --data-dir data/train_pairs
        modal run cloud/modal_train.py --data-dir data/train_pairs --max-iters 100000
    """
    import pathlib

    data_dir = os.path.abspath(data_dir)
    input_dir = os.path.join(data_dir, "input")
    target_dir = os.path.join(data_dir, "target")

    if not os.path.isdir(input_dir) or not os.path.isdir(target_dir):
        raise FileNotFoundError(f"Training pairs not found in {data_dir} (need input/ and target/)")

    input_files = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    target_files = sorted(glob.glob(os.path.join(target_dir, "*.png")))
    print(f"Training data: {len(input_files)} inputs, {len(target_files)} targets")

    # Volume paths
    vol_data_dir = f"{VOL_MOUNT}/train_pairs"
    vol_ckpt_dir = f"{VOL_MOUNT}/checkpoints/nafnet_distill"
    vol_pretrained = f"{VOL_MOUNT}/pretrained/NAFNet-SIDD-width64.pth"

    # Upload training data + pretrained weights
    print(f"\nUploading training data to Modal volume...")
    t0 = time.time()

    # Check what's already uploaded by counting (we can't easily check individual files)
    # Upload everything (Modal handles dedup/overwrites efficiently)
    pretrained_local = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..",
        "reference-code", "NAFNet", "experiments", "pretrained_models", "NAFNet-SIDD-width64.pth"
    )

    with vol.batch_upload(force=True) as batch:
        # Upload pretrained weights
        if os.path.exists(pretrained_local):
            batch.put_file(pretrained_local, "/pretrained/NAFNet-SIDD-width64.pth")
            print(f"  Uploading pretrained weights...")

        # Upload training pairs
        for f in input_files:
            fname = os.path.basename(f)
            batch.put_file(f, f"/train_pairs/input/{fname}")
        for f in target_files:
            fname = os.path.basename(f)
            batch.put_file(f, f"/train_pairs/target/{fname}")

    upload_time = time.time() - t0
    print(f"  Upload done in {upload_time:.0f}s")

    # Run training
    print(f"\nStarting training on H100 ({max_iters} iters, crop={crop_size}, bs={batch_size}, perceptual={perceptual_weight})...")
    result_path = train_remote.remote(
        data_dir=vol_data_dir,
        checkpoint_dir=vol_ckpt_dir,
        pretrained_path=vol_pretrained,
        max_iters=max_iters,
        batch_size=batch_size,
        lr=lr,
        loss=loss,
        resume=resume,
        val_freq=val_freq,
        save_freq=save_freq,
        crop_size=crop_size,
        grad_clip=grad_clip,
        perceptual_weight=perceptual_weight,
    )
    print(f"\nTraining complete. Best model at: {result_path}")

    # Download best checkpoint
    local_ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                  "checkpoints", "nafnet_distill")
    os.makedirs(local_ckpt_dir, exist_ok=True)

    for name in ["nafnet_best.pth", "nafnet_final.pth"]:
        vol_path = f"/checkpoints/nafnet_distill/{name}"
        local_path = os.path.join(local_ckpt_dir, name)
        try:
            print(f"Downloading {name}...")
            with open(local_path, "wb") as f:
                for chunk in vol.read_file(vol_path):
                    f.write(chunk)
            size_mb = os.path.getsize(local_path) / 1024**2
            print(f"  Saved: {local_path} ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"  Could not download {name}: {e}")

    print(f"\n{'=' * 60}")
    print(f"DONE. Checkpoints in: {local_ckpt_dir}")
    print(f"{'=' * 60}")
