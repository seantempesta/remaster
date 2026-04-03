"""Paired frame dataset for distillation training.

Loads input/target frame pairs, applies random crops and augmentation.
Supports:
- RAM caching: pre-loads all images as numpy arrays (eliminates disk I/O)
- GPU caching: pre-loads all images as uint8 CUDA tensors (eliminates CPU→GPU transfer,
  does crops+augmentation on GPU — maximizes GPU utilization for small models)
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
        """Pre-load all images as uint8 numpy arrays using parallel I/O."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import psutil

        n = len(self.pairs)
        print(f"  Caching {n} pairs into RAM (parallel)...")
        t0 = time.time()

        def _load_pair(idx):
            inp_path, tgt_path = self.pairs[idx]
            inp = cv2.imread(inp_path, cv2.IMREAD_COLOR)
            tgt = cv2.imread(tgt_path, cv2.IMREAD_COLOR)
            inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
            tgt = cv2.cvtColor(tgt, cv2.COLOR_BGR2RGB)
            return idx, inp, tgt

        # Pre-allocate list
        self.cached_images = [None] * n
        loaded = 0
        n_workers = min(16, os.cpu_count() or 4)

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_load_pair, i): i for i in range(n)}
            for fut in as_completed(futures):
                idx, inp, tgt = fut.result()
                self.cached_images[idx] = (inp, tgt)
                loaded += 1
                if loaded % 200 == 0:
                    mb = psutil.Process().memory_info().rss / 1024**2
                    elapsed = time.time() - t0
                    rate = loaded / elapsed
                    print(f"    {loaded}/{n} loaded ({rate:.0f} pairs/s), "
                          f"RAM: {mb:.0f}MB")

        elapsed = time.time() - t0
        mb = psutil.Process().memory_info().rss / 1024**2
        rate = n / elapsed
        print(f"  Cached {n} pairs in {elapsed:.1f}s ({rate:.0f} pairs/s), "
              f"RAM: {mb:.0f}MB, workers={n_workers}")

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


class GPUCachedDataset:
    """Entire dataset resident on GPU as uint8 tensors.

    Crops + augmentation happen on GPU with torch ops — no CPU involvement,
    no DataLoader, no serialization overhead. For small datasets that fit
    in VRAM (ours: ~16GB for 1300 1080p pairs on 80GB H100).

    Usage:
        gpu_data = GPUCachedDataset("data/train_pairs", crop_size=256, device="cuda")
        for iteration in range(max_iters):
            inp, tgt = gpu_data.sample_batch(batch_size=128)
            # inp, tgt are (B, 3, cs, cs) float16 on GPU, ready for model
    """

    def __init__(self, data_dir, crop_size=256, device="cuda"):
        from concurrent.futures import ThreadPoolExecutor, as_completed

        self.crop_size = crop_size
        self.device = device

        input_dir = os.path.join(data_dir, "input")
        target_dir = os.path.join(data_dir, "target")
        input_files = sorted(glob.glob(os.path.join(input_dir, "*.png")))

        pairs = []
        for inp_path in input_files:
            tgt_path = os.path.join(target_dir, os.path.basename(inp_path))
            if os.path.exists(tgt_path):
                pairs.append((inp_path, tgt_path))

        if not pairs:
            raise FileNotFoundError(f"No matching pairs in {data_dir}")

        # Parallel load from disk → CPU numpy
        print(f"  Loading {len(pairs)} pairs to GPU ({device})...")
        t0 = time.time()

        def _load(idx):
            inp_path, tgt_path = pairs[idx]
            inp = cv2.imread(inp_path, cv2.IMREAD_COLOR)
            tgt = cv2.imread(tgt_path, cv2.IMREAD_COLOR)
            inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
            tgt = cv2.cvtColor(tgt, cv2.COLOR_BGR2RGB)
            return idx, inp, tgt

        cpu_pairs = [None] * len(pairs)
        n_workers = min(16, os.cpu_count() or 4)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futs = {pool.submit(_load, i): i for i in range(len(pairs))}
            done = 0
            for f in as_completed(futs):
                idx, inp, tgt = f.result()
                cpu_pairs[idx] = (inp, tgt)
                done += 1
                if done % 500 == 0:
                    print(f"    {done}/{len(pairs)} loaded from disk...")

        load_time = time.time() - t0
        print(f"  Disk load: {load_time:.1f}s ({len(pairs)/load_time:.0f} pairs/s)")

        # Transfer to GPU as uint8 tensors (HWC format)
        t1 = time.time()
        self.inputs = []
        self.targets = []
        for inp_np, tgt_np in cpu_pairs:
            self.inputs.append(torch.from_numpy(inp_np).to(device))
            self.targets.append(torch.from_numpy(tgt_np).to(device))
        del cpu_pairs

        xfer_time = time.time() - t1
        vram_gb = torch.cuda.memory_allocated(device) / 1024**3
        self.n = len(self.inputs)
        self.h, self.w = self.inputs[0].shape[:2]
        print(f"  GPU transfer: {xfer_time:.1f}s, {self.n} pairs, "
              f"VRAM: {vram_gb:.1f}GB")

    def sample_batch(self, batch_size):
        """Sample a batch with random crops + augmentation, entirely on GPU.

        Returns (inp, tgt) as (B, 3, cs, cs) float32 tensors on GPU.
        """
        cs = self.crop_size
        # Random indices
        indices = torch.randint(0, self.n, (batch_size,))

        # Random crop coordinates
        tops = torch.randint(0, self.h - cs, (batch_size,))
        lefts = torch.randint(0, self.w - cs, (batch_size,))

        # Crop each sample (GPU tensor slicing)
        inp_crops = []
        tgt_crops = []
        for i in range(batch_size):
            idx = indices[i].item()
            t = tops[i].item()
            l = lefts[i].item()
            inp_crops.append(self.inputs[idx][t:t+cs, l:l+cs])
            tgt_crops.append(self.targets[idx][t:t+cs, l:l+cs])

        # Stack → (B, H, W, 3) uint8, then → (B, 3, H, W) float32
        inp_batch = torch.stack(inp_crops).permute(0, 3, 1, 2).float() / 255.0
        tgt_batch = torch.stack(tgt_crops).permute(0, 3, 1, 2).float() / 255.0

        # Batch augmentation on GPU (same transform for input + target)
        # Random horizontal flip (per-sample)
        hflip_mask = torch.rand(batch_size, device=self.device) < 0.5
        if hflip_mask.any():
            inp_batch[hflip_mask] = inp_batch[hflip_mask].flip(3)
            tgt_batch[hflip_mask] = tgt_batch[hflip_mask].flip(3)

        # Random vertical flip (per-sample)
        vflip_mask = torch.rand(batch_size, device=self.device) < 0.5
        if vflip_mask.any():
            inp_batch[vflip_mask] = inp_batch[vflip_mask].flip(2)
            tgt_batch[vflip_mask] = tgt_batch[vflip_mask].flip(2)

        # Random 90° rotation (per-sample, applied after stacking)
        rot_k = torch.randint(0, 4, (batch_size,))
        for k in (1, 2, 3):
            mask = rot_k == k
            if mask.any():
                inp_batch[mask] = torch.rot90(inp_batch[mask], k, [2, 3])
                tgt_batch[mask] = torch.rot90(tgt_batch[mask], k, [2, 3])

        return inp_batch, tgt_batch
