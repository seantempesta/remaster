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
stop_dict = modal.Dict.from_name("train-signals", create_if_missing=True)
VOL_MOUNT = "/mnt/data"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.11.0",
        "torchvision",
        extra_index_url="https://download.pytorch.org/whl/cu126",
    )
    .pip_install(
        "opencv-python-headless",
        "numpy",
        "psutil",
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
    val_dir: str = "",
    width: int = 64,
    middle_blk_num: int = 12,
    enc_blk_nums: str = "2,2,4,8",
    dec_blk_nums: str = "2,2,2,2",
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

    if val_dir:
        n_val = len(glob.glob(os.path.join(val_dir, "input", "*.png")))
        print(f"Validation data: {n_val} frames from {val_dir}")

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Build args
    from train_nafnet import parse_args, train
    import argparse

    class Args:
        pass
    args = Args()
    args.data_dir = data_dir
    args.val_dir = val_dir or None
    args.crop_size = crop_size
    args.batch_size = batch_size
    args.num_workers = 8
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
    args.perceptual_freq = 1  # every iteration
    args.cache_in_ram = True
    args.amp = True
    args.checkpoint_dir = checkpoint_dir
    args.print_freq = 10
    args.val_freq = val_freq
    args.save_freq = save_freq
    args.width = width
    args.middle_blk_num = middle_blk_num
    args.enc_blk_nums = enc_blk_nums
    args.dec_blk_nums = dec_blk_nums
    args.device = "cuda"

    # Graceful stop: check Modal Dict every 50 iters for a stop signal
    stop_key = os.path.basename(checkpoint_dir)  # e.g. "nafnet_w32_mid4"
    def check_stop():
        try:
            return stop_dict.get(stop_key, False)
        except Exception:
            return False
    args.stop_check = check_stop
    args.stop_check_freq = 50

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
    val_dir: str = "data/val_pairs",
    width: int = 64,
    middle_blk_num: int = 12,
    enc_blk_nums: str = "2,2,4,8",
    dec_blk_nums: str = "2,2,2,2",
    checkpoint_dir: str = "",
):
    """
    Upload training data and run NAFNet distillation training on Modal A100.

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

    # Validation data (separate from training)
    val_dir = os.path.abspath(val_dir)
    val_input_files = sorted(glob.glob(os.path.join(val_dir, "input", "*.png"))) if os.path.isdir(val_dir) else []
    val_target_files = sorted(glob.glob(os.path.join(val_dir, "target", "*.png"))) if os.path.isdir(val_dir) else []
    has_val = len(val_input_files) > 0 and len(val_target_files) > 0
    if has_val:
        print(f"Validation data: {len(val_input_files)} pairs from {val_dir}")
    else:
        print(f"WARNING: No validation data found at {val_dir}, validating on training data")

    # Volume paths — derive from dir basenames so different configs work
    data_name = os.path.basename(os.path.abspath(data_dir))
    val_name = os.path.basename(os.path.abspath(val_dir))
    vol_data_dir = f"{VOL_MOUNT}/{data_name}"
    vol_val_dir = f"{VOL_MOUNT}/{val_name}"

    if checkpoint_dir:
        vol_ckpt_dir = f"{VOL_MOUNT}/{checkpoint_dir}"
    else:
        vol_ckpt_dir = f"{VOL_MOUNT}/checkpoints/nafnet_distill"

    pretrained_filename = f"NAFNet-SIDD-width{width}.pth"
    vol_pretrained = f"{VOL_MOUNT}/pretrained/{pretrained_filename}"

    # Upload training data + validation data + pretrained weights
    print(f"\nUploading data to Modal volume...")
    t0 = time.time()

    pretrained_local = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..",
        "reference-code", "NAFNet", "experiments", "pretrained_models", pretrained_filename
    )

    with vol.batch_upload(force=True) as batch:
        # Upload pretrained weights
        if os.path.exists(pretrained_local):
            batch.put_file(pretrained_local, f"/pretrained/{pretrained_filename}")
            print(f"  Uploading pretrained weights ({pretrained_filename})...")

        # Upload training pairs
        for f in input_files:
            fname = os.path.basename(f)
            batch.put_file(f, f"/{data_name}/input/{fname}")
        for f in target_files:
            fname = os.path.basename(f)
            batch.put_file(f, f"/{data_name}/target/{fname}")

        # Upload validation pairs
        if has_val:
            for f in val_input_files:
                fname = os.path.basename(f)
                batch.put_file(f, f"/{val_name}/input/{fname}")
            for f in val_target_files:
                fname = os.path.basename(f)
                batch.put_file(f, f"/{val_name}/target/{fname}")

    upload_time = time.time() - t0
    print(f"  Upload done in {upload_time:.0f}s")

    # Run training
    print(f"\nStarting training on A100 ({max_iters} iters, crop={crop_size}, bs={batch_size}, "
          f"w={width}, mid={middle_blk_num}, perceptual={perceptual_weight})...")
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
        val_dir=vol_val_dir if has_val else "",
        width=width,
        middle_blk_num=middle_blk_num,
        enc_blk_nums=enc_blk_nums,
        dec_blk_nums=dec_blk_nums,
    )
    print(f"\nTraining complete. Best model at: {result_path}")

    # Download best checkpoint
    # Derive local checkpoint dir from volume checkpoint dir
    ckpt_rel = vol_ckpt_dir.replace(VOL_MOUNT + "/", "")
    local_ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ckpt_rel)
    os.makedirs(local_ckpt_dir, exist_ok=True)

    for name in ["nafnet_best.pth", "nafnet_final.pth"]:
        vol_path = f"/{ckpt_rel}/{name}"
        local_path = os.path.join(local_ckpt_dir, name)
        try:
            print(f"Downloading {name}...")
            with open(local_path, "wb") as f:
                vol.read_file_into_fileobj(vol_path, f)
            size_mb = os.path.getsize(local_path) / 1024**2
            print(f"  Saved: {local_path} ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"  Could not download {name}: {e}")

    print(f"\n{'=' * 60}")
    print(f"DONE. Checkpoints in: {local_ckpt_dir}")
    print(f"{'=' * 60}")
