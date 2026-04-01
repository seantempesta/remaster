import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

"""Option 1: Real-ESRGAN roundtrip denoiser — 4x upscale then downscale back to 1080p.
The model removes compression artifacts as a side effect of learned upscaling."""
from lib.paths import DATA_DIR

import os, glob, time
import cv2
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import imageio_ffmpeg

data_dir = str(DATA_DIR)
input_dir = os.path.join(data_dir, 'frames_1080p_src')
output_dir = os.path.join(data_dir, 'frames_1080p_realesrgan_denoise')
os.makedirs(output_dir, exist_ok=True)

print("Loading Real-ESRGAN x4plus...")
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
upsampler = RealESRGANer(
    scale=4,
    model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
    model=model,
    tile=384,
    tile_pad=10,
    pre_pad=0,
    half=True,
    gpu_id=0,
)

frames = sorted(glob.glob(os.path.join(input_dir, '*.png')))[:150]
print(f"Processing {len(frames)} frames (4x up -> downscale to 1080p)...")

start = time.time()
for i, frame_path in enumerate(frames):
    img = cv2.imread(frame_path, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    output, _ = upsampler.enhance(img, outscale=4)
    output = cv2.resize(output, (w, h), interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(os.path.join(output_dir, f'frame_{i+1:05d}.png'), output)
    if i == 0:
        print(f"  First frame: {time.time()-start:.1f}s")
    if (i + 1) % 25 == 0:
        elapsed = time.time() - start
        fps = (i + 1) / elapsed
        print(f"  [{i+1}/{len(frames)}] {fps:.2f} fps, ETA: {(len(frames)-i-1)/fps:.0f}s")

elapsed = time.time() - start
print(f"\nDone! {len(frames)} frames in {elapsed:.1f}s ({len(frames)/elapsed:.2f} fps)")

print("Encoding video...")
ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
import subprocess
subprocess.run([
    ffmpeg, '-y', '-framerate', '23.976',
    '-i', os.path.join(output_dir, 'frame_%05d.png'),
    '-c:v', 'libx264', '-crf', '14', '-pix_fmt', 'yuv420p',
    os.path.join(data_dir, 'clip_1080p_realesrgan_denoise.mp4')
], check=True, capture_output=True)
print(f"Output: {os.path.join(data_dir, 'clip_1080p_realesrgan_denoise.mp4')}")
