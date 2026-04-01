import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

"""SCUNet on the mid-episode clip — direct inference (no tiling) for speed."""
from lib.paths import add_scunet_to_path, resolve_scunet_dir, DATA_DIR
add_scunet_to_path()

import os, glob, time
import numpy as np
import cv2
import torch
from models.network_scunet import SCUNet as net
import imageio_ffmpeg

DEVICE = 'cuda'
MAX_FRAMES = 150

data_dir = str(DATA_DIR)
input_dir = os.path.join(data_dir, 'frames_mid_1080p')
output_dir = os.path.join(data_dir, 'frames_mid_scunet')
MODEL_NAME = 'scunet_color_real_psnr'
model_path = str(resolve_scunet_dir() / "model_zoo" / f'{MODEL_NAME}.pth')

# Clear old output
os.makedirs(output_dir, exist_ok=True)
for f in glob.glob(os.path.join(output_dir, '*.png')):
    os.remove(f)

print(f"Loading SCUNet ({MODEL_NAME})...")
model = net(in_nc=3, config=[4,4,4,4,4,4,4], dim=64)
model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True), strict=True)
model.eval()
for p in model.parameters():
    p.requires_grad = False
model = model.to(DEVICE).half()
print(f"  Model VRAM: {torch.cuda.memory_allocated()/1024**2:.0f}MB")

frames = sorted(glob.glob(os.path.join(input_dir, '*.png')))[:MAX_FRAMES]
print(f"Processing {len(frames)} frames — DIRECT fp16 inference...")

torch.cuda.reset_peak_memory_stats()
start = time.time()
for i, frame_path in enumerate(frames):
    img = cv2.imread(frame_path, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_t = torch.from_numpy(img_rgb.transpose(2, 0, 1)).half().unsqueeze(0) / 255.0
    img_t = img_t.to(DEVICE)

    with torch.no_grad():
        out_t = model(img_t)

    out = (out_t.squeeze(0).clamp(0, 1) * 255).byte().cpu().numpy().transpose(1, 2, 0)
    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_dir, f'frame_{i+1:05d}.png'), out_bgr)

    del img_t, out_t

    if i == 0:
        print(f"  Frame 1: {time.time()-start:.2f}s, peak VRAM: {torch.cuda.max_memory_allocated()/1024**2:.0f}MB")
    if (i + 1) % 25 == 0:
        elapsed = time.time() - start
        fps = (i + 1) / elapsed
        print(f"  [{i+1}/{len(frames)}] {fps:.2f} fps, ETA: {(len(frames)-i-1)/fps:.0f}s")

elapsed = time.time() - start
print(f"\nDONE: {len(frames)} frames in {elapsed:.1f}s ({len(frames)/elapsed:.2f} fps)")
print(f"Peak VRAM: {torch.cuda.max_memory_allocated()/1024**2:.0f}MB")

# Encode video
print("\nEncoding video...")
ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
import subprocess
out_video = os.path.join(data_dir, 'clip_mid_scunet.mp4')
subprocess.run([
    ffmpeg, '-y', '-framerate', '23.976',
    '-i', os.path.join(output_dir, 'frame_%05d.png'),
    '-c:v', 'libx264', '-crf', '14', '-pix_fmt', 'yuv420p',
    out_video
], check=True, capture_output=True)
print(f"Output: {out_video}")
print(f"Original: {os.path.join(data_dir, 'clip_mid_1080p.mp4')}")
