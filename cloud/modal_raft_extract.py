"""
Modal script -- extract raw RAFT alignment data (flows, warped frames, occlusion masks).

Runs RAFT-Things and/or RAFT-Sintel at FULL 1080p resolution on L40S (48GB VRAM).
Saves all intermediate data to disk for local experimentation without re-running RAFT.

Saved per center frame t and each neighbor n (window=4 -> 8 neighbors):
  flow_fwd_{t:06d}_to_{n:06d}.npy   -- forward flow [2, H, W] float16
  flow_bwd_{t:06d}_to_{n:06d}.npy   -- backward flow [2, H, W] float16
  warped_{t:06d}_from_{n:06d}.png    -- warped neighbor aligned to center (uint8 PNG)
  mask_{t:06d}_from_{n:06d}.npy      -- occlusion mask [H, W] bool

Also saves original frames:
  original_{t:06d}.png

Results written incrementally -- if it crashes, we keep what we have.

Usage:
    PYTHONUTF8=1 modal run cloud/modal_raft_extract.py \
        --input data/archive/firefly_s01e08_30s.mkv \
        --models things,sintel --max-frames 30

    # Skip re-uploading (data already on volume)
    PYTHONUTF8=1 modal run cloud/modal_raft_extract.py \
        --input data/archive/firefly_s01e08_30s.mkv \
        --models things --max-frames 30 --skip-upload
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
)

app = modal.App("remaster-raft-extract", image=image)

RAFT_MODELS = {
    "things": "raft-things.pth",
    "sintel": "raft-sintel.pth",
    "chairs": "raft-chairs.pth",
    "kitti": "raft-kitti.pth",
    "small": "raft-small.pth",
}

# Forward-backward consistency threshold (pixels)
FB_CONSISTENCY_THRESH = 1.5

# RAFT iterations -- 20 is the default
RAFT_ITERS = 20


@app.function(
    gpu="L40S",
    volumes={VOL_MOUNT: vol},
    timeout=14400,  # 4 hours
    memory=32768,   # 32GB RAM for frame storage
)
def extract_raft_data(
    model_name: str,
    video_vol_path: str,
    output_vol_path: str,
    weights_vol_path: str,
    window: int = 4,
    max_frames: int = 0,
):
    """Extract raw RAFT alignment data on cloud GPU.

    For each center frame, computes forward/backward flow to all neighbors,
    warps neighbors, computes occlusion masks. Saves everything incrementally.
    """
    import subprocess
    import sys

    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F

    sys.path.insert(0, "/root/project")
    sys.path.insert(0, "/root/project/reference-code/RAFT/core")

    from raft import RAFT
    from utils.utils import InputPadder

    vol.reload()

    video_path = os.path.join(VOL_MOUNT, video_vol_path)
    weights_path = os.path.join(VOL_MOUNT, weights_vol_path)
    output_dir = os.path.join(VOL_MOUNT, output_vol_path)

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"RAFT weights not found: {weights_path}")

    # Create output subdirs
    originals_dir = os.path.join(output_dir, "originals")
    warped_dir = os.path.join(output_dir, "warped")
    flows_dir = os.path.join(output_dir, "flows")
    masks_dir = os.path.join(output_dir, "masks")
    for d in [originals_dir, warped_dir, flows_dir, masks_dir]:
        os.makedirs(d, exist_ok=True)

    print(f"[extract] Model: RAFT-{model_name.capitalize()}", flush=True)
    print(f"[extract] Video: {video_path}", flush=True)
    print(f"[extract] Output: {output_dir}", flush=True)
    print(f"[extract] Window: {window} ({2 * window} neighbors per frame)", flush=True)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[extract] GPU: {gpu_name} ({gpu_mem:.1f} GB)", flush=True)

    # -----------------------------------------------------------------------
    # Load RAFT model
    # -----------------------------------------------------------------------
    is_small = (model_name == "small")

    class RAFTArgs:
        def __init__(self):
            self.small = is_small
            self.mixed_precision = True
            self.alternate_corr = False
            self.dropout = 0.0

        def __contains__(self, key):
            return hasattr(self, key)

    model = torch.nn.DataParallel(RAFT(RAFTArgs()))
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model = model.module
    model.to("cuda")
    model.eval()
    print(f"[extract] RAFT model loaded", flush=True)

    # -----------------------------------------------------------------------
    # Extract frames from video
    # -----------------------------------------------------------------------
    from lib.ffmpeg_utils import get_video_info

    w, h, fps, total_frames, duration = get_video_info(video_path)
    if max_frames > 0:
        total_frames = min(total_frames, max_frames)

    print(f"[extract] Video: {w}x{h} @ {fps:.3f} fps, extracting {total_frames} frames", flush=True)

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", video_path,
        "-pix_fmt", "rgb24",
        "-f", "rawvideo",
    ]
    if max_frames > 0:
        cmd += ["-frames:v", str(total_frames)]
    cmd += ["pipe:1"]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    frame_size = w * h * 3
    frames = []
    while True:
        raw = proc.stdout.read(frame_size)
        if len(raw) < frame_size:
            break
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
        frames.append(frame)
        if max_frames > 0 and len(frames) >= total_frames:
            break

    proc.stdout.close()
    proc.wait()
    n_frames = len(frames)
    print(f"[extract] Got {n_frames} frames ({n_frames * frame_size / 1e9:.2f} GB RAM)", flush=True)

    # -----------------------------------------------------------------------
    # Save original frames
    # -----------------------------------------------------------------------
    print(f"[extract] Saving original frames...", flush=True)
    for i, frame in enumerate(frames):
        path = os.path.join(originals_dir, f"original_{i:06d}.png")
        # Convert RGB to BGR for cv2
        cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    vol.commit()
    print(f"[extract] Saved {n_frames} originals", flush=True)

    # -----------------------------------------------------------------------
    # Helper functions (inline to avoid import issues)
    # -----------------------------------------------------------------------
    def numpy_to_raft_tensor(frame_np):
        """HWC uint8 numpy -> 1CHW float32 GPU tensor (0-255)."""
        return torch.from_numpy(frame_np).permute(2, 0, 1).float().unsqueeze(0).cuda()

    @torch.no_grad()
    def compute_flow(frame1_np, frame2_np):
        """Full-resolution flow, no downscaling. Returns [1, 2, H, W] on CPU."""
        img1 = numpy_to_raft_tensor(frame1_np)
        img2 = numpy_to_raft_tensor(frame2_np)

        padder = InputPadder(img1.shape)
        img1_p, img2_p = padder.pad(img1, img2)

        _, flow_up = model(img1_p, img2_p, iters=RAFT_ITERS, test_mode=True)
        flow_up = padder.unpad(flow_up)

        flow_cpu = flow_up.cpu()
        del img1, img2, img1_p, img2_p, flow_up
        return flow_cpu

    def warp_frame(frame_np, flow_cpu):
        """Warp frame using flow via grid_sample. Returns (warped_np, in_bounds)."""
        H, W = frame_np.shape[:2]

        gy, gx = torch.meshgrid(
            torch.arange(H, dtype=torch.float32),
            torch.arange(W, dtype=torch.float32),
            indexing="ij",
        )
        flow_sq = flow_cpu.squeeze(0)  # [2, H, W]
        sample_x = gx + flow_sq[0]
        sample_y = gy + flow_sq[1]

        norm_x = 2.0 * sample_x / (W - 1) - 1.0
        norm_y = 2.0 * sample_y / (H - 1) - 1.0
        grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0)

        frame_t = torch.from_numpy(frame_np).permute(2, 0, 1).float().unsqueeze(0)

        warped = F.grid_sample(
            frame_t, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        warped = warped.squeeze(0).permute(1, 2, 0).numpy()  # [H, W, 3]

        in_bounds = (
            (sample_x >= 0) & (sample_x <= W - 1) &
            (sample_y >= 0) & (sample_y <= H - 1)
        ).numpy()

        return warped, in_bounds

    def compute_occlusion_mask(flow_fwd_cpu, flow_bwd_cpu):
        """Forward-backward consistency check. Returns [H, W] bool (True = valid)."""
        H, W = flow_fwd_cpu.shape[2], flow_fwd_cpu.shape[3]

        gy, gx = torch.meshgrid(
            torch.arange(H, dtype=torch.float32),
            torch.arange(W, dtype=torch.float32),
            indexing="ij",
        )
        flow_fwd = flow_fwd_cpu.squeeze(0)
        target_x = gx + flow_fwd[0]
        target_y = gy + flow_fwd[1]

        norm_x = 2.0 * target_x / (W - 1) - 1.0
        norm_y = 2.0 * target_y / (H - 1) - 1.0
        grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0)

        flow_bwd_sampled = F.grid_sample(
            flow_bwd_cpu, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )

        round_trip = flow_fwd_cpu + flow_bwd_sampled
        error = torch.norm(round_trip, dim=1).squeeze(0)

        valid_mask = (error < FB_CONSISTENCY_THRESH).numpy()
        return valid_mask

    # -----------------------------------------------------------------------
    # Process each center frame
    # -----------------------------------------------------------------------
    total_pairs = 0
    t_start = time.time()
    commit_interval = 5  # commit volume every N center frames

    for center_idx in range(n_frames):
        t_frame = time.time()

        # Get neighbor indices within window
        neighbors = []
        for offset in range(-window, window + 1):
            if offset == 0:
                continue
            n_idx = center_idx + offset
            if 0 <= n_idx < n_frames:
                neighbors.append(n_idx)

        if not neighbors:
            continue

        center_frame = frames[center_idx]
        n_pairs_this_frame = 0

        for n_idx in neighbors:
            neighbor_frame = frames[n_idx]

            # Compute forward flow: center -> neighbor
            flow_fwd = compute_flow(center_frame, neighbor_frame)

            # Compute backward flow: neighbor -> center
            flow_bwd = compute_flow(neighbor_frame, center_frame)

            # Warp neighbor to center's grid
            warped, in_bounds = warp_frame(neighbor_frame, flow_fwd)

            # Occlusion mask via forward-backward consistency
            valid_mask = compute_occlusion_mask(flow_fwd, flow_bwd)
            # Combine with in-bounds check
            final_mask = valid_mask & in_bounds

            # Save flow fields as float16
            fwd_path = os.path.join(
                flows_dir, f"flow_fwd_{center_idx:06d}_to_{n_idx:06d}.npy"
            )
            bwd_path = os.path.join(
                flows_dir, f"flow_bwd_{center_idx:06d}_to_{n_idx:06d}.npy"
            )
            np.save(fwd_path, flow_fwd.squeeze(0).numpy().astype(np.float16))
            np.save(bwd_path, flow_bwd.squeeze(0).numpy().astype(np.float16))

            # Save warped frame as PNG (uint8)
            warped_uint8 = np.clip(warped, 0, 255).astype(np.uint8)
            warped_path = os.path.join(
                warped_dir, f"warped_{center_idx:06d}_from_{n_idx:06d}.png"
            )
            cv2.imwrite(warped_path, cv2.cvtColor(warped_uint8, cv2.COLOR_RGB2BGR))

            # Save occlusion mask as bool
            mask_path = os.path.join(
                masks_dir, f"mask_{center_idx:06d}_from_{n_idx:06d}.npy"
            )
            np.save(mask_path, final_mask)

            n_pairs_this_frame += 1
            total_pairs += 1

            # Clean up GPU memory
            del flow_fwd, flow_bwd
            torch.cuda.empty_cache()

        elapsed_frame = time.time() - t_frame
        elapsed_total = time.time() - t_start
        fps_est = (center_idx + 1) / elapsed_total if elapsed_total > 0 else 0
        remaining = (n_frames - center_idx - 1) / fps_est if fps_est > 0 else 0

        print(
            f"[extract] Frame {center_idx + 1}/{n_frames}: "
            f"{n_pairs_this_frame} pairs in {elapsed_frame:.1f}s | "
            f"Total: {total_pairs} pairs | "
            f"ETA: {remaining / 60:.1f} min",
            flush=True,
        )

        # Commit volume periodically
        if (center_idx + 1) % commit_interval == 0:
            vol.commit()

    # Final commit
    vol.commit()

    elapsed_total = time.time() - t_start
    print(
        f"\n[extract] Done: {total_pairs} pairs from {n_frames} frames "
        f"in {elapsed_total / 60:.1f} min",
        flush=True,
    )
    return total_pairs


@app.local_entrypoint()
def main(
    input: str = "data/archive/firefly_s01e08_30s.mkv",
    models: str = "things,sintel",
    window: int = 4,
    max_frames: int = 30,
    skip_upload: bool = False,
):
    """
    Extract raw RAFT alignment data on Modal cloud GPU.

    Uploads video + RAFT weights, runs extraction, downloads all results.

    Examples:
        PYTHONUTF8=1 modal run cloud/modal_raft_extract.py \\
            --input data/archive/firefly_s01e08_30s.mkv \\
            --models things,sintel --max-frames 30
    """
    import pathlib

    input_path = os.path.abspath(input)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")

    model_list = [m.strip().lower() for m in models.split(",")]
    for m in model_list:
        if m not in RAFT_MODELS:
            raise ValueError(
                f"Unknown model '{m}'. Choose from: {', '.join(RAFT_MODELS.keys())}"
            )

    video_stem = pathlib.Path(input_path).stem
    local_output_base = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "archive", "raft_modal_extract",
    )

    # Collect weight files
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
    vol_video_path = f"raft_extract/{video_stem}.mkv"
    vol_weights_dir = "raft_extract/weights"

    # Upload video + weights
    if not skip_upload:
        print(f"Uploading video and weights to Modal volume...")
        t0 = time.time()

        with vol.batch_upload(force=True) as batch:
            batch.put_file(input_path, f"/{vol_video_path}")
            print(
                f"  Video: {os.path.basename(input_path)} "
                f"({os.path.getsize(input_path) / 1e6:.1f} MB)"
            )

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
    print(f"\nExtracting with {len(model_list)} model(s): {', '.join(model_list)}")
    print(f"  Window: {window} ({2 * window} neighbors per center frame)")
    print(f"  Max frames: {max_frames if max_frames > 0 else 'all'}")
    print()

    results = {}
    for m in model_list:
        vol_output = f"raft_extract/output/{video_stem}_{m}"
        vol_weights = f"{vol_weights_dir}/{RAFT_MODELS[m]}"

        print(f"{'=' * 60}")
        print(f"Running RAFT-{m.capitalize()} extraction...")
        print(f"{'=' * 60}")
        t0 = time.time()

        n_pairs = extract_raft_data.remote(
            model_name=m,
            video_vol_path=vol_video_path,
            output_vol_path=vol_output,
            weights_vol_path=vol_weights,
            window=window,
            max_frames=max_frames,
        )

        elapsed = time.time() - t0
        results[m] = {"n_pairs": n_pairs, "elapsed": elapsed}
        print(
            f"  RAFT-{m.capitalize()}: {n_pairs} pairs in {elapsed / 60:.1f} min"
        )

    # Download results
    print(f"\n{'=' * 60}")
    print("Downloading results...")
    print(f"{'=' * 60}")

    subdirs = ["originals", "warped", "flows", "masks"]

    for m in model_list:
        vol_output = f"raft_extract/output/{video_stem}_{m}"
        local_model_dir = os.path.join(local_output_base, m)

        counts = {}
        for subdir in subdirs:
            local_subdir = os.path.join(local_model_dir, subdir)
            os.makedirs(local_subdir, exist_ok=True)

            count = 0
            try:
                for entry in vol.listdir(f"{vol_output}/{subdir}/"):
                    fname = entry.path.split("/")[-1]
                    if not (fname.endswith(".png") or fname.endswith(".npy")):
                        continue
                    local_file = os.path.join(local_subdir, fname)
                    with open(local_file, "wb") as f:
                        vol.read_file_into_fileobj(
                            f"/{vol_output}/{subdir}/{fname}", f
                        )
                    count += 1
            except Exception as e:
                print(f"  Error downloading {subdir} for {m}: {e}")

            counts[subdir] = count

        print(
            f"  RAFT-{m.capitalize()}: "
            f"{counts.get('originals', 0)} originals, "
            f"{counts.get('warped', 0)} warped, "
            f"{counts.get('flows', 0)} flows, "
            f"{counts.get('masks', 0)} masks "
            f"-> {local_model_dir}"
        )

    # Summary
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    total_cost = 0
    for m, info in results.items():
        cost = info["elapsed"] / 3600 * 1.95  # L40S rate
        total_cost += cost
        print(
            f"  RAFT-{m.capitalize()}: {info['n_pairs']} pairs, "
            f"{info['elapsed'] / 60:.1f} min, ~${cost:.2f}"
        )
    print(f"  Total estimated cost: ~${total_cost:.2f}")
    print(f"  Output: {local_output_base}")
    print(f"{'=' * 60}")
