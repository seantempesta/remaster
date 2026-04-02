"""Paired frame dataset for distillation training.

Loads input/target frame pairs, applies random crops and augmentation.
Supports optional RAM caching for eliminating DataLoader bottlenecks
on cloud GPUs with plenty of memory.
"""
import os
import glob
import time
import random

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


class PairedFrameDataset(Dataset):
    """
    Loads paired input/target frames, returns random crops with augmentation.

    Directory structure:
        data_dir/input/frame_XXXXX.png   (compressed input)
        data_dir/target/frame_XXXXX.png  (SCUNet denoised)
    """
    def __init__(self, data_dir, crop_size=256, augment=True,
                 max_frames=-1, cache_in_ram=False):
        self.crop_size = crop_size
        self.augment = augment
        self.cache_in_ram = cache_in_ram
        self.cached_images = None

        input_dir = os.path.join(data_dir, "input")
        target_dir = os.path.join(data_dir, "target")

        input_files = sorted(glob.glob(os.path.join(input_dir, "*.png")))
        if not input_files:
            raise FileNotFoundError(f"No PNG files in {input_dir}")

        # Match input/target pairs by filename
        self.pairs = []
        for inp_path in input_files:
            fname = os.path.basename(inp_path)
            tgt_path = os.path.join(target_dir, fname)
            if os.path.exists(tgt_path):
                self.pairs.append((inp_path, tgt_path))

        if max_frames > 0:
            self.pairs = self.pairs[:max_frames]

        if not self.pairs:
            raise FileNotFoundError(
                f"No matching pairs found in {data_dir}")

        if cache_in_ram:
            self._load_all_into_ram()

        print(f"Dataset: {len(self.pairs)} pairs, crop={crop_size}, "
              f"augment={augment}, cached={cache_in_ram}")

    def _load_all_into_ram(self):
        """Pre-load all images as uint8 numpy arrays."""
        import psutil
        print(f"  Caching {len(self.pairs)} pairs into RAM...")
        t0 = time.time()
        self.cached_images = []
        for i, (inp_path, tgt_path) in enumerate(self.pairs):
            inp = cv2.imread(inp_path, cv2.IMREAD_COLOR)
            tgt = cv2.imread(tgt_path, cv2.IMREAD_COLOR)
            inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
            tgt = cv2.cvtColor(tgt, cv2.COLOR_BGR2RGB)
            # Store as uint8 to save RAM (convert to float in __getitem__)
            self.cached_images.append((inp, tgt))
            if (i + 1) % 200 == 0:
                mb = psutil.Process().memory_info().rss / 1024**2
                print(f"    {i+1}/{len(self.pairs)} loaded, "
                      f"RAM: {mb:.0f}MB")
        elapsed = time.time() - t0
        mb = psutil.Process().memory_info().rss / 1024**2
        print(f"  Cached {len(self.pairs)} pairs in {elapsed:.1f}s, "
              f"RAM: {mb:.0f}MB")

    def __len__(self):
        return len(self.pairs) * 100

    def __getitem__(self, idx):
        pair_idx = idx % len(self.pairs)

        if self.cached_images is not None:
            inp, tgt = self.cached_images[pair_idx]
            inp = inp.astype(np.float32) / 255.0
            tgt = tgt.astype(np.float32) / 255.0
        else:
            inp_path, tgt_path = self.pairs[pair_idx]
            inp = cv2.imread(inp_path, cv2.IMREAD_COLOR)
            tgt = cv2.imread(tgt_path, cv2.IMREAD_COLOR)
            inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(
                np.float32) / 255.0
            tgt = cv2.cvtColor(tgt, cv2.COLOR_BGR2RGB).astype(
                np.float32) / 255.0

        h, w, _ = inp.shape
        cs = self.crop_size

        # Random crop (same location for both)
        top = random.randint(0, h - cs)
        left = random.randint(0, w - cs)
        inp = inp[top:top + cs, left:left + cs]
        tgt = tgt[top:top + cs, left:left + cs]

        # Augmentation: random flip and rotation
        if self.augment:
            if random.random() < 0.5:
                inp = inp[:, ::-1, :].copy()
                tgt = tgt[:, ::-1, :].copy()
            if random.random() < 0.5:
                inp = inp[::-1, :, :].copy()
                tgt = tgt[::-1, :, :].copy()
            k = random.randint(0, 3)
            if k > 0:
                inp = np.rot90(inp, k).copy()
                tgt = np.rot90(tgt, k).copy()

        inp = torch.from_numpy(inp.transpose(2, 0, 1))
        tgt = torch.from_numpy(tgt.transpose(2, 0, 1))
        return inp, tgt
