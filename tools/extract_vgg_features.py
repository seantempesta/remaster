"""Extract VGG16 feature vectors from images using DISTS-style stages.

Produces a 1472-dim feature vector per image (global avg pool of VGG stages 1-5,
channels: 64+128+256+512+512 = 1472, skipping the 3-channel input passthrough).

Usage:
    python tools/extract_vgg_features.py --input-dir data/originals --output data/originals/vgg_features.pkl
"""

import argparse
import os
import sys
import pickle
import time
import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# VGG16 feature extractor (stages from DISTS) with L2 pooling
# ---------------------------------------------------------------------------

class L2pooling(nn.Module):
    def __init__(self, filter_size=5, stride=2, channels=None, pad_off=0):
        super().__init__()
        self.padding = (filter_size - 2) // 2
        self.stride = stride
        self.channels = channels
        a = np.hanning(filter_size)[1:-1]
        g = torch.Tensor(a[:, None] * a[None, :])
        g = g / torch.sum(g)
        self.register_buffer('filter', g[None, None, :, :].repeat((self.channels, 1, 1, 1)))

    def forward(self, input):
        input = input ** 2
        out = F.conv2d(input, self.filter, stride=self.stride, padding=self.padding, groups=input.shape[1])
        return (out + 1e-12).sqrt()


def build_vgg16_stages():
    """Build VGG16 feature stages using torchvision."""
    from torchvision import models
    vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features

    stage1 = nn.Sequential()
    stage2 = nn.Sequential()
    stage3 = nn.Sequential()
    stage4 = nn.Sequential()
    stage5 = nn.Sequential()

    for x in range(0, 4):
        stage1.add_module(str(x), vgg[x])
    stage2.add_module(str(4), L2pooling(channels=64))
    for x in range(5, 9):
        stage2.add_module(str(x), vgg[x])
    stage3.add_module(str(9), L2pooling(channels=128))
    for x in range(10, 16):
        stage3.add_module(str(x), vgg[x])
    stage4.add_module(str(16), L2pooling(channels=256))
    for x in range(17, 23):
        stage4.add_module(str(x), vgg[x])
    stage5.add_module(str(23), L2pooling(channels=512))
    for x in range(24, 30):
        stage5.add_module(str(x), vgg[x])

    return stage1, stage2, stage3, stage4, stage5


class VGGFeatureExtractor(nn.Module):
    """Extracts 1472-dim feature vector (global avg pool of VGG stages 1-5)."""

    def __init__(self):
        super().__init__()
        self.stage1, self.stage2, self.stage3, self.stage4, self.stage5 = build_vgg16_stages()
        for param in self.parameters():
            param.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1))

    @torch.no_grad()
    def forward(self, x):
        """x: (1, 3, H, W) float tensor in [0, 1]. Returns (1472,) numpy array."""
        h = (x - self.mean) / self.std
        h = self.stage1(h)
        f1 = h.mean([2, 3]).squeeze(0)  # 64
        h = self.stage2(h)
        f2 = h.mean([2, 3]).squeeze(0)  # 128
        h = self.stage3(h)
        f3 = h.mean([2, 3]).squeeze(0)  # 256
        h = self.stage4(h)
        f4 = h.mean([2, 3]).squeeze(0)  # 512
        h = self.stage5(h)
        f5 = h.mean([2, 3]).squeeze(0)  # 512
        return torch.cat([f1, f2, f3, f4, f5], dim=0)  # 1472


def load_image_tensor(path, max_size=512):
    """Load PNG as (1, 3, H, W) float32 tensor, resized so shortest side = max_size."""
    from PIL import Image
    img = Image.open(path).convert('RGB')
    w, h = img.size
    # Resize shortest side to max_size
    if min(w, h) > max_size:
        scale = max_size / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    # Convert to tensor
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor


def main():
    parser = argparse.ArgumentParser(description='Extract VGG16 features from images')
    parser.add_argument('--input-dir', type=str, default='data/originals',
                        help='Directory containing PNG files')
    parser.add_argument('--output', type=str, default='data/originals/vgg_features.pkl',
                        help='Output pickle file')
    parser.add_argument('--max-size', type=int, default=512,
                        help='Resize shortest side to this (default: 512)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size (1 is safest for VRAM)')
    args = parser.parse_args()

    # Find all PNGs
    pattern = os.path.join(args.input_dir, '*.png')
    files = sorted(glob.glob(pattern))
    print(f'Found {len(files)} PNG files in {args.input_dir}')
    if not files:
        print('No files found, exiting.')
        return

    # Load model
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    model = VGGFeatureExtractor().to(device).eval()
    if device.type == 'cuda':
        model = model.half()
    print('VGG16 feature extractor loaded')

    # Extract features
    features = {}
    t0 = time.time()
    for i, fpath in enumerate(files):
        fname = os.path.basename(fpath)
        tensor = load_image_tensor(fpath, max_size=args.max_size).to(device)
        if device.type == 'cuda':
            tensor = tensor.half()
        feat = model(tensor).float().cpu().numpy()
        features[fname] = feat

        if (i + 1) % 200 == 0 or (i + 1) == len(files):
            elapsed = time.time() - t0
            fps = (i + 1) / elapsed
            eta = (len(files) - i - 1) / fps
            print(f'  [{i+1}/{len(files)}] {fps:.1f} img/s, ETA {eta:.0f}s')

    # Save
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)

    total = time.time() - t0
    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f'Done: {len(features)} features saved to {args.output} ({size_mb:.1f} MB) in {total:.0f}s')


if __name__ == '__main__':
    main()
