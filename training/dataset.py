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
        """Pre-load random crops into RAM instead of full images.

        Caches CROPS_PER_IMAGE random crops per pair as uint8 numpy arrays.
        At 256x256x3, each crop pair is ~400KB vs ~12MB for full 1080p.
        Crops are refreshed every epoch via refresh_cache().

        Memory: 6K pairs x 4 crops x 400KB = ~10GB (vs ~80GB for full images).
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import psutil

        n = len(self.pairs)
        self._crops_per_image = 4
        CHUNK_SIZE = 200
        n_workers = min(8, os.cpu_count() or 4)
        total_crops = n * self._crops_per_image
        print(f"  Caching {n} pairs x {self._crops_per_image} crops = "
              f"{total_crops} crop pairs into RAM...")
        t0 = time.time()

        cs = self.crop_size

        def _load_crops(idx):
            inp_path, tgt_path = self.pairs[idx]
            inp = cv2.imread(inp_path, cv2.IMREAD_COLOR)
            tgt = cv2.imread(tgt_path, cv2.IMREAD_COLOR)
            if inp is None:
                raise IOError(f"Failed to read {inp_path}")
            if tgt is None:
                raise IOError(f"Failed to read {tgt_path}")
            inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
            tgt = cv2.cvtColor(tgt, cv2.COLOR_BGR2RGB)
            h, w = inp.shape[:2]

            crops = []
            for _ in range(self._crops_per_image):
                top = random.randint(0, h - cs)
                left = random.randint(0, w - cs)
                crops.append((
                    inp[top:top + cs, left:left + cs].copy(),
                    tgt[top:top + cs, left:left + cs].copy(),
                ))
            return idx, crops

        self.cached_images = [None] * total_crops
        loaded = 0

        for chunk_start in range(0, n, CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, n)
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_load_crops, i): i
                           for i in range(chunk_start, chunk_end)}
                for fut in as_completed(futures):
                    idx, crops = fut.result()
                    for c, (ic, tc) in enumerate(crops):
                        self.cached_images[idx * self._crops_per_image + c] = (ic, tc)
                    loaded += 1
            if loaded % 200 == 0 or chunk_end == n:
                mb = psutil.Process().memory_info().rss / 1024**2
                elapsed = time.time() - t0
                rate = loaded / elapsed if elapsed > 0 else 0
                print(f"    {loaded}/{n} images loaded ({rate:.0f} img/s), "
                      f"RAM: {mb:.0f}MB")

        elapsed = time.time() - t0
        mb = psutil.Process().memory_info().rss / 1024**2
        rate = n / elapsed if elapsed > 0 else 0
        print(f"  Cached {total_crops} crops in {elapsed:.1f}s ({rate:.0f} img/s), "
              f"RAM: {mb:.0f}MB")

    def refresh_cache(self):
        """Re-generate random crops from disk. Call between epochs for variety."""
        if self.cached_images is not None and self.cache_in_ram:
            print("  Refreshing crop cache...")
            self._load_all_into_ram()

    def __len__(self):
        if self.cached_images is not None:
            return len(self.cached_images) * 25  # 25 augmentations per crop
        return len(self.pairs) * 100

    def __getitem__(self, idx):
        if self.cached_images is not None:
            # Index into pre-cropped cache
            crop_idx = idx % len(self.cached_images)
            inp, tgt = self.cached_images[crop_idx]
            inp = inp.astype(np.float32) / 255.0
            tgt = tgt.astype(np.float32) / 255.0
        else:
            # Uncached: load from disk and random crop
            pair_idx = idx % len(self.pairs)
            inp_path, tgt_path = self.pairs[pair_idx]
            inp = cv2.imread(inp_path, cv2.IMREAD_COLOR)
            tgt = cv2.imread(tgt_path, cv2.IMREAD_COLOR)
            inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(
                np.float32) / 255.0
            tgt = cv2.cvtColor(tgt, cv2.COLOR_BGR2RGB).astype(
                np.float32) / 255.0

            h, w, _ = inp.shape
            cs = self.crop_size
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


class InputOnlyDataset(Dataset):
    """
    Loads only input frames (no targets needed) for online teacher distillation.

    Directory structure:
        data_dir/input/frame_XXXXX.png   (compressed input)

    Returns (input_crop, input_crop) — target slot is a dummy copy of input.
    The caller is expected to override the target with teacher output.
    """
    def __init__(self, data_dir, crop_size=256, augment=True,
                 max_frames=-1, cache_in_ram=False):
        self.crop_size = crop_size
        self.augment = augment
        self.cache_in_ram = cache_in_ram
        self.cached_images = None

        input_dir = os.path.join(data_dir, "input")
        input_files = sorted(glob.glob(os.path.join(input_dir, "*.png")))
        if not input_files:
            raise FileNotFoundError(f"No PNG files in {input_dir}")

        self.files = input_files
        if max_frames > 0:
            self.files = self.files[:max_frames]

        if cache_in_ram:
            self._load_all_into_ram()

        print(f"InputOnlyDataset: {len(self.files)} frames, crop={crop_size}, "
              f"augment={augment}, cached={cache_in_ram}")

    def _load_all_into_ram(self):
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import psutil

        n = len(self.files)
        CHUNK_SIZE = 200
        n_workers = min(8, os.cpu_count() or 4)
        print(f"  Caching {n} input frames into RAM "
              f"(chunks of {CHUNK_SIZE}, {n_workers} workers)...")
        t0 = time.time()

        def _load(idx):
            img = cv2.imread(self.files[idx], cv2.IMREAD_COLOR)
            if img is None:
                raise IOError(f"Failed to read {self.files[idx]}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return idx, img

        self.cached_images = [None] * n
        loaded = 0

        for chunk_start in range(0, n, CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, n)
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_load, i): i
                           for i in range(chunk_start, chunk_end)}
                for fut in as_completed(futures):
                    idx, img = fut.result()
                    self.cached_images[idx] = img
                    loaded += 1
            if loaded % 200 == 0 or chunk_end == n:
                mb = psutil.Process().memory_info().rss / 1024**2
                elapsed = time.time() - t0
                rate = loaded / elapsed if elapsed > 0 else 0
                print(f"    {loaded}/{n} loaded ({rate:.0f} pairs/s), "
                      f"RAM: {mb:.0f}MB")

        elapsed = time.time() - t0
        mb = psutil.Process().memory_info().rss / 1024**2
        print(f"  Cached {n} frames in {elapsed:.1f}s, RAM: {mb:.0f}MB")

    def __len__(self):
        return len(self.files) * 100

    def __getitem__(self, idx):
        file_idx = idx % len(self.files)

        if self.cached_images is not None:
            inp = self.cached_images[file_idx].astype(np.float32) / 255.0
        else:
            inp = cv2.imread(self.files[file_idx], cv2.IMREAD_COLOR)
            inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        h, w, _ = inp.shape
        cs = self.crop_size

        top = random.randint(0, h - cs)
        left = random.randint(0, w - cs)
        inp = inp[top:top + cs, left:left + cs]

        if self.augment:
            if random.random() < 0.5:
                inp = inp[:, ::-1, :].copy()
            if random.random() < 0.5:
                inp = inp[::-1, :, :].copy()
            k = random.randint(0, 3)
            if k > 0:
                inp = np.rot90(inp, k).copy()

        inp = torch.from_numpy(inp.transpose(2, 0, 1))
        # Return inp twice — caller overrides second with teacher output
        return inp, inp.clone()


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

        # Chunked parallel load from disk -> CPU numpy
        CHUNK_SIZE = 200
        n_workers = min(8, os.cpu_count() or 4)
        print(f"  Loading {len(pairs)} pairs to GPU ({device}), "
              f"chunks of {CHUNK_SIZE}...")
        t0 = time.time()

        def _load(idx):
            inp_path, tgt_path = pairs[idx]
            inp = cv2.imread(inp_path, cv2.IMREAD_COLOR)
            tgt = cv2.imread(tgt_path, cv2.IMREAD_COLOR)
            if inp is None:
                raise IOError(f"Failed to read {inp_path}")
            if tgt is None:
                raise IOError(f"Failed to read {tgt_path}")
            inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
            tgt = cv2.cvtColor(tgt, cv2.COLOR_BGR2RGB)
            return idx, inp, tgt

        cpu_pairs = [None] * len(pairs)
        done = 0
        for chunk_start in range(0, len(pairs), CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, len(pairs))
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futs = {pool.submit(_load, i): i
                        for i in range(chunk_start, chunk_end)}
                for f in as_completed(futs):
                    idx, inp, tgt = f.result()
                    cpu_pairs[idx] = (inp, tgt)
                    done += 1
            if done % 200 == 0 or chunk_end == len(pairs):
                print(f"    {done}/{len(pairs)} loaded from disk...")

        load_time = time.time() - t0
        print(f"  Disk load: {load_time:.1f}s "
              f"({len(pairs)/load_time:.0f} pairs/s)")

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
