"""Test SDPA replacement in SCUNet attention — verify correctness + measure speedup."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.paths import add_scunet_to_path, resolve_scunet_dir
add_scunet_to_path()

import time
import torch
import torch.nn.functional as F
import numpy as np
from models.network_scunet import SCUNet as net, WMSA

DEVICE = 'cuda'
model_path = str(resolve_scunet_dir() / 'model_zoo' / 'scunet_color_real_psnr.pth')

# Monkey-patch WMSA.forward with SDPA version
original_forward = WMSA.forward

def sdpa_forward(self, x):
    from einops import rearrange
    if self.type != 'W':
        x = torch.roll(x, shifts=(-(self.window_size//2), -(self.window_size//2)), dims=(1,2))
    x = rearrange(x, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)
    h_windows = x.size(1)
    w_windows = x.size(2)
    x = rearrange(x, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)

    qkv = self.embedding_layer(x)
    # (batch, n_windows, n_patches, 3*dim) -> split into q,k,v
    # Original: rearrange to (heads, batch, windows, patches, head_dim) then einsum
    # SDPA needs: (batch*windows, heads, patches, head_dim)
    B, NW, NP, _ = qkv.shape
    qkv = qkv.reshape(B, NW, NP, 3, self.n_heads, self.head_dim)
    q, k, v = qkv[..., 0, :, :], qkv[..., 1, :, :], qkv[..., 2, :, :]
    # -> (B*NW, heads, NP, head_dim)
    q = q.reshape(B * NW, NP, self.n_heads, self.head_dim).transpose(1, 2)
    k = k.reshape(B * NW, NP, self.n_heads, self.head_dim).transpose(1, 2)
    v = v.reshape(B * NW, NP, self.n_heads, self.head_dim).transpose(1, 2)

    # Relative position bias: (heads, NP, NP) -> (1, heads, NP, NP) -> broadcast
    rel_bias = self.relative_embedding().unsqueeze(0).expand(B * NW, -1, -1, -1)

    # Attention mask for shifted windows
    if self.type != 'W':
        attn_mask = self.generate_mask(h_windows, w_windows, self.window_size, shift=self.window_size//2)
        # (1, 1, NW, NP, NP) -> (B*NW, 1, NP, NP) to broadcast over heads
        attn_mask = attn_mask.reshape(NW, NP, NP).unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        attn_mask = attn_mask.repeat(B, 1, 1, 1)  # (B*NW, heads, NP, NP)
        rel_bias = rel_bias.clone()
        rel_bias.masked_fill_(attn_mask, float("-inf"))

    output = F.scaled_dot_product_attention(q, k, v, attn_mask=rel_bias)
    # (B*NW, heads, NP, head_dim) -> (B, NW, NP, dim)
    output = output.transpose(1, 2).reshape(B, NW, NP, -1)
    output = self.linear(output)
    output = rearrange(output, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1=h_windows, p1=self.window_size)

    if self.type != 'W':
        output = torch.roll(output, shifts=(self.window_size//2, self.window_size//2), dims=(1,2))
    return output


# Load model with ORIGINAL attention
model_orig = net(in_nc=3, config=[4,4,4,4,4,4,4], dim=64)
model_orig.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True), strict=True)
model_orig.eval().to(DEVICE).half()
for p in model_orig.parameters():
    p.requires_grad = False

# Test correctness with a small input first
print("Testing correctness on 256x256...")
test_input = torch.randn(1, 3, 256, 256, device=DEVICE, dtype=torch.float16)
with torch.no_grad():
    out_orig = model_orig(test_input)

# Patch and test
WMSA.forward = sdpa_forward
model_sdpa = net(in_nc=3, config=[4,4,4,4,4,4,4], dim=64)
model_sdpa.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True), strict=True)
model_sdpa.eval().to(DEVICE).half()
for p in model_sdpa.parameters():
    p.requires_grad = False

with torch.no_grad():
    out_sdpa = model_sdpa(test_input)

diff = (out_orig.float() - out_sdpa.float()).abs().max().item()
print(f"  Max difference: {diff:.6f} (should be < 0.01)")

if diff > 0.1:
    print("  ERROR: outputs differ too much, SDPA patch is broken!")
    sys.exit(1)

print("  PASS — outputs match")

# Benchmark both at 1080p
del model_orig, test_input, out_orig, out_sdpa
torch.cuda.empty_cache()

dummy = torch.randn(1, 3, 1080, 1920, device=DEVICE, dtype=torch.float16)

# Benchmark SDPA version
print("\nBenchmarking SDPA at 1080p...")
torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    # warmup
    out = model_sdpa(dummy); del out
    torch.cuda.synchronize()
    times = []
    for _ in range(3):
        t0 = time.time()
        out = model_sdpa(dummy)
        torch.cuda.synchronize()
        times.append(time.time() - t0)
        del out

avg = np.mean(times)
peak = torch.cuda.max_memory_allocated() / 1024**2
print(f"  SDPA: {avg:.2f}s, {1/avg:.2f} fps, peak VRAM: {peak:.0f}MB")

# Compare with original
del model_sdpa
torch.cuda.empty_cache()
WMSA.forward = original_forward
model_orig2 = net(in_nc=3, config=[4,4,4,4,4,4,4], dim=64)
model_orig2.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True), strict=True)
model_orig2.eval().to(DEVICE).half()
for p in model_orig2.parameters():
    p.requires_grad = False

print("\nBenchmarking original at 1080p...")
torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    out = model_orig2(dummy); del out
    torch.cuda.synchronize()
    times = []
    for _ in range(3):
        t0 = time.time()
        out = model_orig2(dummy)
        torch.cuda.synchronize()
        times.append(time.time() - t0)
        del out

avg2 = np.mean(times)
peak2 = torch.cuda.max_memory_allocated() / 1024**2
print(f"  Original: {avg2:.2f}s, {1/avg2:.2f} fps, peak VRAM: {peak2:.0f}MB")

print(f"\nSpeedup: {avg2/avg:.2f}x")
