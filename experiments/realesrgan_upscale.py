import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

"""Baseline: Real-ESRGAN frame-by-frame upscale from 480p to 1080p."""
from lib.paths import DATA_DIR

import os, glob, time
import cv2
import torch
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

data_dir = str(DATA_DIR)
input_dir = os.path.join(data_dir, 'frames_480p')
output_dir = os.path.join(data_dir, 'frames_realesrgan')
os.makedirs(output_dir, exist_ok=True)

print("Loading Real-ESRGAN model...")
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
upsampler = RealESRGANer(
    scale=4,
    model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
    model=model,
    tile=384,
    tile_pad=10,
    pre_pad=0,
    half=False,
    gpu_id=0,
)
print("Model loaded.")

frames = sorted(glob.glob(os.path.join(input_dir, '*.png')))
MAX_FRAMES = 150  # ~6 seconds, enough for comparison; set to len(frames) for full clip
frames = frames[:MAX_FRAMES]
print(f"Processing {len(frames)} frames with Real-ESRGAN...")
print(f"Input: 854x480, Output: 1920x1080 (4x then resize)")

start = time.time()
for i, frame_path in enumerate(frames):
    img = cv2.imread(frame_path, cv2.IMREAD_COLOR)
    output, _ = upsampler.enhance(img, outscale=4)
    # Resize to exact 1080p for comparison
    output = cv2.resize(output, (1920, 1080), interpolation=cv2.INTER_LANCZOS4)
    out_path = os.path.join(output_dir, os.path.basename(frame_path))
    cv2.imwrite(out_path, output)
    if i == 0:
        print(f"  First frame done in {time.time()-start:.1f}s")
    if (i + 1) % 50 == 0:
        elapsed = time.time() - start
        fps = (i + 1) / elapsed
        eta = (len(frames) - i - 1) / fps
        print(f"  [{i+1}/{len(frames)}] {fps:.2f} fps, ETA: {eta:.0f}s")

elapsed = time.time() - start
print(f"\nDone! {len(frames)} frames in {elapsed:.1f}s ({len(frames)/elapsed:.2f} fps)")
