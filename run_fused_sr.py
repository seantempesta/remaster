"""Test A: Upscale the flow-fused frames with Real-ESRGAN."""
import os, glob, time
import cv2
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

data_dir = r'C:\Users\sean\src\upscale-experiment\data'
input_dir = os.path.join(data_dir, 'frames_fused_480p')
output_dir = os.path.join(data_dir, 'frames_warp_fuse_sr')
os.makedirs(output_dir, exist_ok=True)

print("Loading Real-ESRGAN...")
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

frames = sorted(glob.glob(os.path.join(input_dir, '*.png')))
print(f"Upscaling {len(frames)} fused frames...")

start = time.time()
for i, frame_path in enumerate(frames):
    img = cv2.imread(frame_path, cv2.IMREAD_COLOR)
    output, _ = upsampler.enhance(img, outscale=4)
    output = cv2.resize(output, (1920, 1080), interpolation=cv2.INTER_LANCZOS4)
    out_path = os.path.join(output_dir, os.path.basename(frame_path))
    cv2.imwrite(out_path, output)
    if i == 0:
        print(f"  First frame: {time.time()-start:.1f}s")
    if (i + 1) % 50 == 0:
        elapsed = time.time() - start
        fps = (i + 1) / elapsed
        print(f"  [{i+1}/{len(frames)}] {fps:.2f} fps, ETA: {(len(frames)-i-1)/fps:.0f}s")

print(f"\nDone! {len(frames)} frames in {time.time()-start:.1f}s")
