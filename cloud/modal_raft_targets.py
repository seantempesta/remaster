"""
Modal wrapper — run RAFT temporal target generation on cloud GPU.

Uses full-size RAFT models (Things, Sintel) at full 1080p resolution,
which require more VRAM than the local RTX 3060 can provide.

Usage:
    # Run with RAFT-Things and RAFT-Sintel on the same clip
    modal run cloud/modal_raft_targets.py \
        --input data/archive/firefly_s01e08_30s.mkv \
        --models things,sintel

    # Single model, custom settings
    modal run cloud/modal_raft_targets.py \
        --input data/archive/firefly_s01e08_30s.mkv \
        --models things --window 6 --max-frames 50

    # Use cheaper A10G instead of L40S
    modal run cloud/modal_raft_targets.py \
        --input data/archive/firefly_s01e08_30s.mkv \
        --models sintel --gpu A10G
"""
import modal
import os
import time

vol = modal.Volume.from_name("upscale-data", create_if_missing=True)
VOL_MOUNT = "/mnt/data"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1", "libglib2.0-0", "ffmpeg")
    .pip_install(
        "torch==2.11.0",
        "torchvision",
        extra_index_url="https://download.pytorch.org/whl/cu126",
    )
    .pip_install(
        "opencv-python-headless",
        "numpy",
        "pillow",
        "scipy",
    )
    # RAFT core source files
    .add_local_file("reference-code/RAFT/core/raft.py", remote_path="/root/project/reference-code/RAFT/core/raft.py")
    .add_local_file("reference-code/RAFT/core/corr.py", remote_path="/root/project/reference-code/RAFT/core/corr.py")
    .add_local_file("reference-code/RAFT/core/extractor.py", remote_path="/root/project/reference-code/RAFT/core/extractor.py")
    .add_local_file("reference-code/RAFT/core/update.py", remote_path="/root/project/reference-code/RAFT/core/update.py")
    .add_local_file("reference-code/RAFT/core/__init__.py", remote_path="/root/project/reference-code/RAFT/core/__init__.py")
    .add_local_file("reference-code/RAFT/core/utils/__init__.py", remote_path="/root/project/reference-code/RAFT/core/utils/__init__.py")
    .add_local_file("reference-code/RAFT/core/utils/utils.py", remote_path="/root/project/reference-code/RAFT/core/utils/utils.py")
    .add_local_file("reference-code/RAFT/core/utils/flow_viz.py", remote_path="/root/project/reference-code/RAFT/core/utils/flow_viz.py")
    .add_local_file("reference-code/RAFT/core/utils/frame_utils.py", remote_path="/root/project/reference-code/RAFT/core/utils/frame_utils.py")
    .add_local_file("reference-code/RAFT/core/utils/augmentor.py", remote_path="/root/project/reference-code/RAFT/core/utils/augmentor.py")
    # lib/ for ffmpeg_utils
    .add_local_file("lib/__init__.py", remote_path="/root/project/lib/__init__.py")
    .add_local_file("lib/ffmpeg_utils.py", remote_path="/root/project/lib/ffmpeg_utils.py")
    # The main processing script
    .add_local_file("tools/raft_temporal_targets.py", remote_path="/root/project/tools/raft_temporal_targets.py")
)

app = modal.App("remaster-raft-targets", image=image)

# Map of model names to weight filenames
RAFT_MODELS = {
    "things": "raft-things.pth",
    "sintel": "raft-sintel.pth",
    "chairs": "raft-chairs.pth",
    "kitti": "raft-kitti.pth",
    "small": "raft-small.pth",
}


@app.function(
    gpu="L40S",
    volumes={VOL_MOUNT: vol},
    timeout=14400,  # 4 hours
    memory=32768,   # 32GB RAM for frame storage
)
def run_raft_targets(
    model_name: str,
    video_vol_path: str,
    output_vol_path: str,
    weights_vol_path: str,
    window: int = 4,
    max_frames: int = 0,
    comparison: bool = True,
    recursive: bool = True,
):
    """Run RAFT temporal target generation on cloud GPU."""
    import sys
    sys.path.insert(0, "/root/project")
    sys.path.insert(0, "/root/project/reference-code/RAFT/core")

    vol.reload()

    # Verify files exist
    video_path = os.path.join(VOL_MOUNT, video_vol_path)
    weights_path = os.path.join(VOL_MOUNT, weights_vol_path)
    output_dir = os.path.join(VOL_MOUNT, output_vol_path)

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"RAFT weights not found: {weights_path}")

    os.makedirs(output_dir, exist_ok=True)

    print(f"[modal] RAFT model: {model_name}", flush=True)
    print(f"[modal] Video: {video_path}", flush=True)
    print(f"[modal] Output: {output_dir}", flush=True)
    print(f"[modal] Weights: {weights_path}", flush=True)
    print(f"[modal] Window: {window}, recursive: {recursive}", flush=True)

    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[modal] GPU: {gpu_name} ({gpu_mem:.1f} GB)", flush=True)

    # Patch RAFT paths and globals to use our weights and full resolution
    import tools.raft_temporal_targets as raft_tool

    # Override the model paths to point at volume
    if model_name == "small":
        raft_tool.RAFT_SMALL_PATH = weights_path
    else:
        raft_tool.RAFT_THINGS_PATH = weights_path

    # Full resolution -- no downscaling, we have plenty of VRAM
    raft_tool.FLOW_HALF_RES = False
    raft_tool.USE_SMALL = (model_name == "small")

    # Override compute_flow_pair to always use full resolution on cloud
    # (the default auto-scales to 0.75 for 1080p, but we have 48GB VRAM)
    @torch.no_grad()
    def compute_flow_full_res(model, frame1_np, frame2_np, prefilter=False,
                              prefilter_sigma=2.0):
        """Full-resolution flow -- no downscaling on cloud GPU.

        Returns flow on GPU (not CPU) for the GPU-accelerated pipeline.
        """
        from utils.utils import InputPadder

        if prefilter:
            frame1_np = raft_tool.prefilter_for_flow(frame1_np, sigma=prefilter_sigma)
            frame2_np = raft_tool.prefilter_for_flow(frame2_np, sigma=prefilter_sigma)

        img1 = raft_tool.numpy_to_raft_tensor(frame1_np)
        img2 = raft_tool.numpy_to_raft_tensor(frame2_np)

        padder = InputPadder(img1.shape)
        img1_p, img2_p = padder.pad(img1, img2)

        _, flow_up = model(img1_p, img2_p, iters=raft_tool.RAFT_ITERS, test_mode=True)
        flow_up = padder.unpad(flow_up)

        flow_cpu = flow_up.cpu()
        del img1, img2, img1_p, img2_p, flow_up
        torch.cuda.empty_cache()

        return flow_cpu

    raft_tool.compute_flow_pair = compute_flow_full_res

    # Run processing
    use_small = (model_name == "small")
    raft_tool.process_video(
        input_path=video_path,
        output_dir=output_dir,
        window=window,
        max_frames=max_frames if max_frames > 0 else None,
        save_comparison=comparison,
        small=use_small,
        recursive=recursive,
    )

    # Count output files
    targets_dir = os.path.join(output_dir, "targets")
    n_targets = len([f for f in os.listdir(targets_dir) if f.endswith(".png")]) if os.path.isdir(targets_dir) else 0
    print(f"\n[modal] Done: {n_targets} targets generated for model={model_name}", flush=True)

    vol.commit()
    return n_targets


@app.local_entrypoint()
def main(
    input: str = "data/archive/firefly_s01e08_30s.mkv",
    models: str = "things,sintel",
    window: int = 4,
    max_frames: int = 0,
    comparison: bool = True,
    recursive: bool = True,
    output_base: str = "",
    skip_upload: bool = False,
):
    """
    Run RAFT temporal target generation on Modal cloud GPU.

    Uploads video + RAFT weights, runs processing, downloads results.

    Examples:
        # Compare RAFT-Things vs RAFT-Sintel
        modal run cloud/modal_raft_targets.py \\
            --input data/archive/firefly_s01e08_30s.mkv \\
            --models things,sintel

        # Single model, limited frames
        modal run cloud/modal_raft_targets.py \\
            --input data/archive/firefly_s01e08_30s.mkv \\
            --models things --max-frames 30

        # Skip re-uploading (data already on volume)
        modal run cloud/modal_raft_targets.py \\
            --input data/archive/firefly_s01e08_30s.mkv \\
            --models sintel --skip-upload
    """
    import pathlib
    import glob

    input_path = os.path.abspath(input)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")

    model_list = [m.strip().lower() for m in models.split(",")]
    for m in model_list:
        if m not in RAFT_MODELS:
            raise ValueError(f"Unknown model '{m}'. Choose from: {', '.join(RAFT_MODELS.keys())}")

    # Determine output base directory
    video_stem = pathlib.Path(input_path).stem
    if output_base:
        local_output_base = os.path.abspath(output_base)
    else:
        local_output_base = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "archive", f"raft_{video_stem}",
        )

    # Collect weight files to upload
    raft_models_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "reference-code", "RAFT", "models",
    )
    weights_to_upload = {}
    for m in model_list:
        weight_file = os.path.join(raft_models_dir, RAFT_MODELS[m])
        if not os.path.exists(weight_file):
            raise FileNotFoundError(f"RAFT weights not found: {weight_file}")
        weights_to_upload[m] = weight_file

    # Volume paths
    vol_video_path = f"raft_work/{video_stem}.mkv"
    vol_weights_dir = "raft_work/weights"

    # Upload video + weights to volume
    if not skip_upload:
        print(f"Uploading video and weights to Modal volume...")
        t0 = time.time()

        with vol.batch_upload(force=True) as batch:
            # Upload video
            batch.put_file(input_path, f"/{vol_video_path}")
            print(f"  Video: {os.path.basename(input_path)} ({os.path.getsize(input_path) / 1e6:.1f} MB)")

            # Upload weight files (deduplicated)
            uploaded = set()
            for m, wpath in weights_to_upload.items():
                fname = RAFT_MODELS[m]
                if fname not in uploaded:
                    batch.put_file(wpath, f"/{vol_weights_dir}/{fname}")
                    size_mb = os.path.getsize(wpath) / 1e6
                    print(f"  Weights: {fname} ({size_mb:.1f} MB)")
                    uploaded.add(fname)

        print(f"  Upload done in {time.time() - t0:.1f}s")
    else:
        print("Skipping upload (--skip-upload). Using existing data on volume.")

    # Run each model
    print(f"\nProcessing with {len(model_list)} model(s): {', '.join(model_list)}")
    print(f"  Window: {window} ({2 * window + 1} frames)")
    print(f"  Recursive: {recursive}")
    print(f"  Comparison: {comparison}")
    print(f"  Max frames: {max_frames if max_frames > 0 else 'all'}")
    print()

    results = {}
    for m in model_list:
        vol_output = f"raft_work/output/{video_stem}_{m}"
        vol_weights = f"{vol_weights_dir}/{RAFT_MODELS[m]}"

        print(f"{'=' * 60}")
        print(f"Running RAFT-{m.capitalize()} ...")
        print(f"{'=' * 60}")
        t0 = time.time()

        n_targets = run_raft_targets.remote(
            model_name=m,
            video_vol_path=vol_video_path,
            output_vol_path=vol_output,
            weights_vol_path=vol_weights,
            window=window,
            max_frames=max_frames,
            comparison=comparison,
            recursive=recursive,
        )

        elapsed = time.time() - t0
        results[m] = {"n_targets": n_targets, "elapsed": elapsed}
        print(f"  RAFT-{m.capitalize()}: {n_targets} targets in {elapsed / 60:.1f} min")

    # Download results
    print(f"\n{'=' * 60}")
    print("Downloading results...")
    print(f"{'=' * 60}")

    for m in model_list:
        vol_output = f"raft_work/output/{video_stem}_{m}"
        local_output = os.path.join(local_output_base, m)
        os.makedirs(local_output, exist_ok=True)

        # Download targets
        targets_local = os.path.join(local_output, "targets")
        os.makedirs(targets_local, exist_ok=True)

        target_count = 0
        compare_count = 0

        try:
            for entry in vol.listdir(f"{vol_output}/targets/"):
                fname = entry.path.split("/")[-1]
                if not fname.endswith(".png"):
                    continue
                local_file = os.path.join(targets_local, fname)
                with open(local_file, "wb") as f:
                    vol.read_file_into_fileobj(f"/{vol_output}/targets/{fname}", f)
                target_count += 1
        except Exception as e:
            print(f"  Error downloading targets for {m}: {e}")

        # Download comparisons
        if comparison:
            compare_local = os.path.join(local_output, "comparisons")
            os.makedirs(compare_local, exist_ok=True)
            try:
                for entry in vol.listdir(f"{vol_output}/comparisons/"):
                    fname = entry.path.split("/")[-1]
                    if not fname.endswith(".png"):
                        continue
                    local_file = os.path.join(compare_local, fname)
                    with open(local_file, "wb") as f:
                        vol.read_file_into_fileobj(f"/{vol_output}/comparisons/{fname}", f)
                    compare_count += 1
            except Exception as e:
                print(f"  Error downloading comparisons for {m}: {e}")

        # Download progress file
        try:
            progress_local = os.path.join(local_output, "_progress.json")
            with open(progress_local, "wb") as f:
                vol.read_file_into_fileobj(f"/{vol_output}/_progress.json", f)
        except Exception:
            pass

        print(f"  RAFT-{m.capitalize()}: {target_count} targets, {compare_count} comparisons -> {local_output}")

    # Summary
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    total_cost = 0
    for m, info in results.items():
        # L40S is $1.95/hr
        cost = info["elapsed"] / 3600 * 1.95
        total_cost += cost
        print(f"  RAFT-{m.capitalize()}: {info['n_targets']} targets, "
              f"{info['elapsed'] / 60:.1f} min, ~${cost:.2f}")
    print(f"  Total estimated cost: ~${total_cost:.2f}")
    print(f"  Output: {local_output_base}")
    print(f"{'=' * 60}")
