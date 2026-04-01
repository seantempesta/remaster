import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

"""
SCUNet Denoiser — trained on real-world degradations including compression artifacts.
Robust version with profiling, VRAM monitoring, error handling, and resume support.
"""
from lib.paths import add_scunet_to_path, resolve_scunet_dir, DATA_DIR
add_scunet_to_path()

import os, glob, time
import numpy as np
import cv2
import torch
from models.network_scunet import SCUNet as net
from utils import utils_model
import imageio_ffmpeg

DEVICE = 'cuda'
MAX_FRAMES = 150

data_dir = str(DATA_DIR)
input_dir = os.path.join(data_dir, 'frames_1080p_src')
output_dir = os.path.join(data_dir, 'frames_1080p_scunet')
MODEL_NAME = 'scunet_color_real_psnr'
model_path = str(resolve_scunet_dir() / "model_zoo" / f'{MODEL_NAME}.pth')


def vram_mb():
    return torch.cuda.memory_allocated() / 1024**2


def peak_vram_mb():
    return torch.cuda.max_memory_allocated() / 1024**2


def load_model():
    print(f"Loading SCUNet ({MODEL_NAME})...")
    model = net(in_nc=3, config=[4,4,4,4,4,4,4], dim=64)
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True), strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model = model.to(DEVICE)
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  {params_m:.1f}M parameters, VRAM: {vram_mb():.0f}MB")
    return model


def test_direct_inference(model, sample_frame_path):
    """Test if direct (non-tiled) inference fits in VRAM."""
    print("\nTesting inference mode...")
    img = cv2.imread(sample_frame_path, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    print(f"  Frame size: {w}x{h}")

    img_t = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    img_t = img_t.to(DEVICE)

    torch.cuda.reset_peak_memory_stats()
    try:
        with torch.no_grad():
            out = model(img_t)
        peak = peak_vram_mb()
        print(f"  Direct inference OK — peak VRAM: {peak:.0f}MB")
        del img_t, out
        torch.cuda.empty_cache()
        return peak < 6100  # use direct if it fits in 6GB
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print(f"  Direct inference OOM — will use tiling")
            torch.cuda.empty_cache()
            return False
        raise


def process_frame(model, frame_path, use_tiling):
    """Process a single frame through SCUNet."""
    img = cv2.imread(frame_path, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_t = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    img_t = img_t.to(DEVICE)

    with torch.no_grad():
        if use_tiling:
            out_t = utils_model.test_mode(model, img_t, refield=64, min_size=512, mode=2)
        else:
            out_t = model(img_t)

    out = (out_t.squeeze(0).clamp(0, 1) * 255).byte().cpu().numpy().transpose(1, 2, 0)
    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    del img_t, out_t
    return out_bgr


def main():
    # Clear old output
    os.makedirs(output_dir, exist_ok=True)
    for f in glob.glob(os.path.join(output_dir, '*.png')):
        os.remove(f)

    model = load_model()

    frames = sorted(glob.glob(os.path.join(input_dir, '*.png')))[:MAX_FRAMES]
    print(f"\n{len(frames)} frames to process")

    use_direct = test_direct_inference(model, frames[0])

    # Profile first frame
    print(f"\nProcessing with {'direct' if use_direct else 'tiled'} inference...")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = process_frame(model, frames[0], use_tiling=not use_direct)
    first_frame_time = time.time() - t0
    cv2.imwrite(os.path.join(output_dir, 'frame_00001.png'), out)
    print(f"  Frame 1: {first_frame_time:.2f}s, peak VRAM: {peak_vram_mb():.0f}MB")

    # Process remaining frames
    times = [first_frame_time]
    errors = []
    start = time.time()

    for i in range(1, len(frames)):
        t0 = time.time()
        try:
            out = process_frame(model, frames[i], use_tiling=not use_direct)
            cv2.imwrite(os.path.join(output_dir, f'frame_{i+1:05d}.png'), out)
            frame_time = time.time() - t0
            times.append(frame_time)
        except RuntimeError as e:
            errors.append((i, str(e)))
            print(f"  ERROR frame {i+1}: {e}")
            torch.cuda.empty_cache()
            # Copy original as fallback
            import shutil
            shutil.copy2(frames[i], os.path.join(output_dir, f'frame_{i+1:05d}.png'))
            continue

        if (i + 1) % 25 == 0:
            elapsed = time.time() - start
            avg_fps = (i + 1) / elapsed
            recent_fps = 25 / sum(times[-25:])
            eta = (len(frames) - i - 1) / avg_fps
            print(f"  [{i+1}/{len(frames)}] avg: {avg_fps:.2f} fps, recent: {recent_fps:.2f} fps, "
                  f"VRAM: {vram_mb():.0f}MB, ETA: {eta:.0f}s")

    # Summary
    elapsed = time.time() - start + first_frame_time
    print(f"\n{'='*60}")
    print(f"DONE: {len(frames)} frames in {elapsed:.1f}s ({len(frames)/elapsed:.2f} fps)")
    print(f"  Avg frame time: {np.mean(times):.2f}s")
    print(f"  Peak VRAM: {peak_vram_mb():.0f}MB")
    if errors:
        print(f"  Errors: {len(errors)} frames failed (used original as fallback)")
    print(f"{'='*60}")

    # Encode video
    print("\nEncoding video...")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    import subprocess
    out_video = os.path.join(data_dir, 'clip_1080p_scunet.mp4')
    subprocess.run([
        ffmpeg, '-y', '-framerate', '23.976',
        '-i', os.path.join(output_dir, 'frame_%05d.png'),
        '-c:v', 'libx264', '-crf', '14', '-pix_fmt', 'yuv420p',
        out_video
    ], check=True, capture_output=True)
    print(f"Output: {out_video}")


if __name__ == '__main__':
    main()
