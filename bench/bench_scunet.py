"""Quick benchmark: test batch sizes and measure fps + VRAM."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.paths import add_scunet_to_path, resolve_scunet_dir
add_scunet_to_path()

import time
import torch
import numpy as np
from models.network_scunet import SCUNet as net

DEVICE = 'cuda'
model_path = str(resolve_scunet_dir() / 'model_zoo' / 'scunet_color_real_psnr.pth')

model = net(in_nc=3, config=[4,4,4,4,4,4,4], dim=64)
model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True), strict=True)
model.eval()
for p in model.parameters():
    p.requires_grad = False
model = model.to(DEVICE).half()

for batch_size in [1, 2]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    dummy = torch.randn(batch_size, 3, 1080, 1920, device=DEVICE, dtype=torch.float16)

    # Warmup
    with torch.no_grad():
        try:
            out = model(dummy)
            del out
        except RuntimeError as e:
            print(f"Batch {batch_size}: OOM — {e}")
            torch.cuda.empty_cache()
            continue

    # Benchmark
    torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(5):
            t0 = time.time()
            out = model(dummy)
            torch.cuda.synchronize()
            times.append(time.time() - t0)
            del out

    avg = np.mean(times)
    fps = batch_size / avg
    peak = torch.cuda.max_memory_allocated() / 1024**2
    print(f"Batch {batch_size}: {avg:.2f}s/batch, {fps:.2f} fps, peak VRAM: {peak:.0f}MB")
    del dummy
