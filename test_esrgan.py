"""Quick test: 1 frame, check GPU usage and speed."""
import time, cv2, torch
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

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

img = cv2.imread(r'C:\Users\sean\src\upscale-experiment\data\frames_480p\frame_00001.png', cv2.IMREAD_COLOR)
print(f"Input shape: {img.shape}")
print(f"GPU mem before: {torch.cuda.memory_allocated()/1024**2:.0f}MB")

t = time.time()
output, _ = upsampler.enhance(img, outscale=4)
print(f"Output shape: {output.shape}")
print(f"Time: {time.time()-t:.1f}s")
print(f"GPU mem after: {torch.cuda.memory_allocated()/1024**2:.0f}MB")
