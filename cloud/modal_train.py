"""
Modal training wrapper — trains any denoising model on cloud GPU.

Supports: NAFNet, PlainDenoise, UNetDenoise, DRUNet (via --arch flag).
Includes: online teacher distillation, feature matching loss, Prodigy optimizer.

Usage:
    # DRUNet student with feature matching distillation (recommended)
    modal run cloud/modal_train.py --arch drunet --nc-list 16,32,64,128 --nb 2 \
        --teacher checkpoints/drunet_teacher/best.pth --teacher-model drunet \
        --feature-matching-weight 0.1 --optimizer prodigy

    # Quick test
    modal run cloud/modal_train.py --arch drunet --max-iters 200 --val-freq 100

    # Resume previous run
    modal run cloud/modal_train.py --resume
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
        "matplotlib",
        "prodigyopt",
    )
    .add_local_file("lib/plainnet_arch.py", remote_path="/root/project/lib/plainnet_arch.py")
    .add_local_file("lib/nafnet_arch.py", remote_path="/root/project/lib/nafnet_arch.py")
    .add_local_file("lib/paths.py", remote_path="/root/project/lib/paths.py")
    .add_local_file("lib/__init__.py", remote_path="/root/project/lib/__init__.py")
    .add_local_file("training/train.py", remote_path="/root/project/training/train.py")
    .add_local_file("training/losses.py", remote_path="/root/project/training/losses.py")
    .add_local_file("training/dataset.py", remote_path="/root/project/training/dataset.py")
    .add_local_file("training/viz.py", remote_path="/root/project/training/viz.py")
    .add_local_file("training/__init__.py", remote_path="/root/project/training/__init__.py")
    .add_local_file("reference-code/DISTS/DISTS_pytorch/DISTS_pt.py", remote_path="/root/project/reference-code/DISTS/DISTS_pytorch/DISTS_pt.py")
    .add_local_file("reference-code/DISTS/DISTS_pytorch/__init__.py", remote_path="/root/project/reference-code/DISTS/DISTS_pytorch/__init__.py")
    .add_local_file("reference-code/DISTS/DISTS_pytorch/weights.pt", remote_path="/root/project/reference-code/DISTS/DISTS_pytorch/weights.pt")
    # KAIR DRUNet architecture + pretrained teacher weights
    .add_local_file("reference-code/KAIR/models/network_unet.py", remote_path="/root/project/reference-code/KAIR/models/network_unet.py")
    .add_local_file("reference-code/KAIR/models/basicblock.py", remote_path="/root/project/reference-code/KAIR/models/basicblock.py")
    .add_local_file("reference-code/KAIR/models/__init__.py", remote_path="/root/project/reference-code/KAIR/models/__init__.py")
    .add_local_file("reference-code/KAIR/model_zoo/drunet_deblocking_color.pth", remote_path="/root/project/reference-code/KAIR/model_zoo/drunet_deblocking_color.pth")
)

app = modal.App("remaster-train", image=image)

@app.function(
    gpu="A10G",    # A10G (~$1.10/hr), H100 (~$3.50/hr), T4 for debug (~$0.20/hr)
    volumes={VOL_MOUNT: vol},
    timeout=28800,
    memory=65536,  # 64GB for RAM cache
)
def train_remote(
    data_dir: str,
    checkpoint_dir: str,
    max_iters: int = 25000,
    batch_size: int = 512,
    lr: float = 1e-3,
    loss: str = "charbonnier",
    resume: bool = False,
    val_freq: int = 1000,
    save_freq: int = 5000,
    crop_size: int = 256,
    grad_clip: float = 1.0,
    perceptual_weight: float = 0.0,
    perceptual_freq: int = 1,
    fft_weight: float = 0.0,
    fft_alpha: float = 1.0,
    val_dir: str = "",
    arch: str = "unet",
    nc: int = 64,
    nb: int = 15,
    nb_enc: str = "2,2",
    nb_dec: str = "2,2",
    nb_mid: int = 2,
    pretrained_path: str = "",
    ema: bool = True,
    ema_decay: float = 0.999,
    intensity_aug: bool = True,
    qat: bool = False,
    sparse: bool = False,
    cache_in_ram: bool = True,
    num_workers: int = 16,
    # Teacher distillation
    teacher_path: str = "",
    teacher_model: str = "drunet_full",
    teacher_noise_level: int = 15,
    # Optimizer
    optimizer_type: str = "adamw",
    d_coef: float = 1.0,
    # DRUNet
    nc_list: str = "16,32,64,128",
    # Feature matching
    feature_matching_weight: float = 0.0,
):
    """Run training on a cloud GPU."""
    import sys
    sys.path.insert(0, "/root/project")

    vol.reload()

    # Verify data
    input_dir = os.path.join(data_dir, "input")
    n_input = len(glob.glob(os.path.join(input_dir, "*.png")))
    if teacher_path:
        print(f"Training data: {n_input} inputs (teacher provides targets)")
        if n_input == 0:
            raise FileNotFoundError(f"No input images in {input_dir}")
    else:
        target_dir = os.path.join(data_dir, "target")
        n_target = len(glob.glob(os.path.join(target_dir, "*.png")))
        print(f"Training data: {n_input} inputs, {n_target} targets")
        if n_input == 0 or n_target == 0:
            raise FileNotFoundError(f"No training pairs in {data_dir}")

    if val_dir:
        n_val = len(glob.glob(os.path.join(val_dir, "input", "*.png")))
        print(f"Validation data: {n_val} frames from {val_dir}")

    os.makedirs(checkpoint_dir, exist_ok=True)

    from training.train import train

    class Args:
        pass
    args = Args()
    args.data_dir = data_dir
    args.val_dir = val_dir or None
    args.crop_size = crop_size
    args.batch_size = batch_size
    args.num_workers = num_workers
    args.pretrained = pretrained_path if pretrained_path and os.path.exists(pretrained_path) else None
    if resume:
        resume_path = os.path.join(checkpoint_dir, "latest.pth")
        if os.path.exists(resume_path):
            args.resume = resume_path
        else:
            args.resume = None
            print("  No checkpoint found to resume from")
    else:
        args.resume = None
    args.max_iters = max_iters
    args.lr = lr
    args.eta_min = 1e-7
    args.weight_decay = 0.0
    args.warmup_iters = min(500, max_iters // 10)
    args.grad_clip = grad_clip
    args.loss = loss
    args.perceptual_weight = perceptual_weight
    args.perceptual_freq = perceptual_freq
    args.fft_weight = fft_weight
    args.fft_alpha = fft_alpha
    # A10G has 24GB VRAM — not enough for GPU cache (dataset is ~16GB)
    # Use RAM cache instead (64GB system RAM on Modal)
    args.cache_on_gpu = False
    args.cache_in_ram = True
    args.amp = True
    args.checkpoint_dir = checkpoint_dir
    args.print_freq = 10
    args.val_freq = val_freq
    args.save_freq = save_freq
    args.device = "cuda"

    # Architecture — train.py uses --model, not --arch
    args.model = arch
    args.nc = nc
    args.nb = nb
    args.nb_enc = nb_enc
    args.nb_dec = nb_dec
    args.nb_mid = nb_mid
    args.full_res = True  # Full resolution processing (default)
    # NAFNet args (needed even if not used, build_model checks them)
    args.width = nc
    args.middle_blk_num = nb_mid
    args.enc_blk_nums = nb_enc
    args.dec_blk_nums = nb_dec

    # DRUNet nc-list
    args.nc_list = nc_list

    # Teacher (online distillation)
    args.teacher = teacher_path if teacher_path else None
    args.teacher_model = teacher_model
    args.teacher_noise_level = teacher_noise_level
    # Teacher architecture args (for building the teacher model)
    args.teacher_width = 64
    args.teacher_middle_blk_num = 12
    args.teacher_enc_blk_nums = '2,2,4,8'
    args.teacher_dec_blk_nums = '2,2,2,2'
    args.teacher_nc_list = '64,128,256,512'
    args.teacher_nb = 4
    args.teacher_nc = 64
    args.teacher_nb_mid = 4

    # Optimizer
    args.optimizer = optimizer_type
    args.d_coef = d_coef

    # Feature matching
    args.feature_matching_weight = feature_matching_weight

    # Enhancements
    args.ema = ema
    args.ema_decay = ema_decay
    args.intensity_aug = intensity_aug
    args.qat = qat
    args.sparse = sparse

    # Graceful stop via Modal Dict
    stop_key = os.path.basename(checkpoint_dir)
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

    best_path = os.path.join(checkpoint_dir, "best.pth")
    if os.path.exists(best_path):
        return best_path
    return os.path.join(checkpoint_dir, "final.pth")


@app.local_entrypoint()
def main(
    data_dir: str = "data/train_pairs",
    max_iters: int = 25000,
    batch_size: int = 512,
    lr: float = 1e-3,
    loss: str = "charbonnier",
    resume: bool = True,
    val_freq: int = 1000,
    save_freq: int = 5000,
    crop_size: int = 256,
    grad_clip: float = 1.0,
    perceptual_weight: float = 0.0,
    perceptual_freq: int = 1,
    fft_weight: float = 0.0,
    fft_alpha: float = 1.0,
    val_dir: str = "data/val_pairs",
    arch: str = "unet",
    nc: int = 64,
    nb: int = 15,
    nb_enc: str = "2,2",
    nb_dec: str = "2,2",
    nb_mid: int = 2,
    checkpoint_dir: str = "",
    pretrained: str = "",
    ema: bool = True,
    ema_decay: float = 0.999,
    intensity_aug: bool = True,
    qat: bool = False,
    sparse: bool = False,
    gpu: str = "T4",
    # Teacher distillation
    teacher: str = "",
    teacher_model: str = "drunet_full",
    teacher_noise_level: int = 15,
    # Optimizer
    optimizer: str = "adamw",
    d_coef: float = 1.0,
    # DRUNet
    nc_list: str = "16,32,64,128",
    # Feature matching
    feature_matching_weight: float = 0.0,
):
    """
    Train denoising models on Modal GPU.

    Supports: NAFNet, PlainDenoise, UNetDenoise, DRUNet (via --arch flag).

    Examples:
        # DRUNet student with feature matching distillation
        modal run cloud/modal_train.py --arch drunet --nc-list 16,32,64,128 \\
            --teacher checkpoints/drunet_teacher/best.pth --teacher-model drunet \\
            --feature-matching-weight 0.1 --optimizer prodigy

        # Debug run on cheap T4
        modal run cloud/modal_train.py --gpu T4 --max-iters 200
    """
    import pathlib

    data_dir = os.path.abspath(data_dir)
    input_dir = os.path.join(data_dir, "input")
    target_dir = os.path.join(data_dir, "target")

    input_files = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    target_files = sorted(glob.glob(os.path.join(target_dir, "*.png"))) if os.path.isdir(target_dir) else []

    if teacher:
        if not input_files:
            raise FileNotFoundError(f"No input images in {input_dir}")
        print(f"Training data: {len(input_files)} inputs (teacher provides targets)")
    else:
        if not input_files or not target_files:
            raise FileNotFoundError(f"Training pairs not found in {data_dir}")
        print(f"Training data: {len(input_files)} inputs, {len(target_files)} targets")

    # Validation data
    val_dir = os.path.abspath(val_dir)
    val_input_files = sorted(glob.glob(os.path.join(val_dir, "input", "*.png"))) if os.path.isdir(val_dir) else []
    val_target_files = sorted(glob.glob(os.path.join(val_dir, "target", "*.png"))) if os.path.isdir(val_dir) else []
    has_val = len(val_input_files) > 0 and len(val_target_files) > 0
    if has_val:
        print(f"Validation data: {len(val_input_files)} pairs from {val_dir}")
    else:
        print(f"WARNING: No validation data at {val_dir}, validating on training data")

    # Volume paths
    data_name = os.path.basename(os.path.abspath(data_dir))
    val_name = os.path.basename(os.path.abspath(val_dir))
    vol_data_dir = f"{VOL_MOUNT}/{data_name}"
    vol_val_dir = f"{VOL_MOUNT}/{val_name}"

    if checkpoint_dir:
        vol_ckpt_dir = f"{VOL_MOUNT}/{checkpoint_dir}"
    else:
        arch_tag = f"{arch}_nc{nc}"
        if arch == "unet":
            arch_tag += f"_mid{nb_mid}"
        else:
            arch_tag += f"_nb{nb}"
        vol_ckpt_dir = f"{VOL_MOUNT}/checkpoints/{arch_tag}"

    # Pretrained weights
    vol_pretrained = ""
    if pretrained and os.path.exists(pretrained):
        pretrained_local = os.path.abspath(pretrained)
        vol_pretrained = f"{VOL_MOUNT}/pretrained/{os.path.basename(pretrained_local)}"

    # Teacher weights (for online distillation)
    vol_teacher = ""
    if teacher and os.path.exists(teacher):
        teacher_local = os.path.abspath(teacher)
        vol_teacher = f"{VOL_MOUNT}/pretrained/{os.path.basename(teacher_local)}"

    # Upload data
    print(f"\nUploading data to Modal volume...")
    t0 = time.time()

    with vol.batch_upload(force=True) as batch:
        if pretrained and os.path.exists(pretrained):
            batch.put_file(os.path.abspath(pretrained),
                          vol_pretrained.replace(VOL_MOUNT, ""))
        if teacher and os.path.exists(teacher):
            batch.put_file(os.path.abspath(teacher),
                          vol_teacher.replace(VOL_MOUNT, ""))
            print(f"  Uploading teacher weights ({os.path.basename(teacher)})...")

        for f in input_files:
            batch.put_file(f, f"/{data_name}/input/{os.path.basename(f)}")
        for f in target_files:
            batch.put_file(f, f"/{data_name}/target/{os.path.basename(f)}")

        if has_val:
            for f in val_input_files:
                batch.put_file(f, f"/{val_name}/input/{os.path.basename(f)}")
            for f in val_target_files:
                batch.put_file(f, f"/{val_name}/target/{os.path.basename(f)}")

    print(f"  Upload done in {time.time()-t0:.0f}s")

    # Run training
    arch_desc = f"{arch} nc={nc} " + (f"mid={nb_mid}" if arch == "unet" else f"nb={nb}")
    print(f"\nStarting training on {gpu}:")
    print(f"  Architecture: {arch_desc}")
    print(f"  Iters: {max_iters}, bs={batch_size}, crop={crop_size}")
    print(f"  Loss: {loss}, perc={perceptual_weight}, fft={fft_weight}")
    print(f"  EMA: {ema}, QAT: {qat}, Sparse: {sparse}")
    print()

    # Note: GPU type is set in @app.function decorator above.
    # Change it there for different tiers (T4 for debug, H100 for production).
    result_path = train_remote.remote(
        data_dir=vol_data_dir,
        checkpoint_dir=vol_ckpt_dir,
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
        perceptual_freq=perceptual_freq,
        fft_weight=fft_weight,
        fft_alpha=fft_alpha,
        val_dir=vol_val_dir if has_val else "",
        arch=arch,
        nc=nc,
        nb=nb,
        nb_enc=nb_enc,
        nb_dec=nb_dec,
        nb_mid=nb_mid,
        pretrained_path=vol_pretrained,
        ema=ema,
        ema_decay=ema_decay,
        intensity_aug=intensity_aug,
        qat=qat,
        sparse=sparse,
        cache_in_ram=True,
        teacher_path=vol_teacher if teacher else "",
        teacher_model=teacher_model,
        teacher_noise_level=teacher_noise_level,
        optimizer_type=optimizer,
        d_coef=d_coef,
        nc_list=nc_list,
        feature_matching_weight=feature_matching_weight,
    )
    print(f"\nTraining complete. Best model at: {result_path}")

    # Download checkpoints
    ckpt_rel = vol_ckpt_dir.replace(VOL_MOUNT + "/", "")
    local_ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ckpt_rel)
    os.makedirs(local_ckpt_dir, exist_ok=True)

    for name in ["best.pth", "final.pth", "training_curves.png",
                  "training_log.json"]:
        vol_path = f"/{ckpt_rel}/{name}"
        local_path = os.path.join(local_ckpt_dir, name)
        try:
            with open(local_path, "wb") as f:
                vol.read_file_into_fileobj(vol_path, f)
            size_mb = os.path.getsize(local_path) / 1024**2
            print(f"  Downloaded {name} ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"  Could not download {name}: {e}")

    # Download sample comparison images
    local_samples_dir = os.path.join(local_ckpt_dir, "samples")
    os.makedirs(local_samples_dir, exist_ok=True)
    try:
        # List sample files on volume
        samples_prefix = f"{ckpt_rel}/samples/"
        sample_count = 0
        for entry in vol.listdir(samples_prefix):
            name = entry.path.split("/")[-1]
            if not name.endswith(".png"):
                continue
            local_path = os.path.join(local_samples_dir, name)
            try:
                with open(local_path, "wb") as f:
                    vol.read_file_into_fileobj(f"/{samples_prefix}{name}", f)
                sample_count += 1
            except Exception:
                pass
        if sample_count > 0:
            print(f"  Downloaded {sample_count} sample images to {local_samples_dir}")
    except Exception as e:
        print(f"  Could not list samples: {e}")

    print(f"\n{'='*60}")
    print(f"DONE. Checkpoints in: {local_ckpt_dir}")
    print(f"{'='*60}")
