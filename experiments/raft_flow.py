import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

"""Compute optical flow between consecutive frames using RAFT.
Saves flow as .npy files and visualizations as PNGs."""
from lib.paths import add_raft_to_path, resolve_raft_dir, DATA_DIR
add_raft_to_path()

import argparse
import os
import glob
import time
import numpy as np
import torch
from PIL import Image
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

DEVICE = 'cuda'

def load_image(path):
    img = np.array(Image.open(path)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def main():
    data_dir = str(DATA_DIR)
    frames_dir = os.path.join(data_dir, 'frames_480p')
    flow_dir = os.path.join(data_dir, 'flow_npy')
    flow_vis_dir = os.path.join(data_dir, 'flow_vis')
    os.makedirs(flow_dir, exist_ok=True)
    os.makedirs(flow_vis_dir, exist_ok=True)
    args = argparse.Namespace(
        model=str(resolve_raft_dir() / "models" / "raft-things.pth"),
        small=False,
        mixed_precision=True,
        alternate_corr=False,
    )
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model, map_location=DEVICE))
    model = model.module
    model.to(DEVICE)
    model.eval()
    images = sorted(glob.glob(os.path.join(frames_dir, '*.png')))
    print(f"Computing optical flow for {len(images)-1} frame pairs...")
    start = time.time()
    with torch.no_grad():
        for i in range(len(images) - 1):
            img1 = load_image(images[i])
            img2 = load_image(images[i + 1])
            padder = InputPadder(img1.shape)
            img1p, img2p = padder.pad(img1, img2)
            _, flow_fwd = model(img1p, img2p, iters=20, test_mode=True)
            flow_fwd = padder.unpad(flow_fwd)
            _, flow_bwd = model(img2p, img1p, iters=20, test_mode=True)
            flow_bwd = padder.unpad(flow_bwd)
            flow_fwd_np = flow_fwd[0].permute(1, 2, 0).cpu().numpy()
            flow_bwd_np = flow_bwd[0].permute(1, 2, 0).cpu().numpy()
            np.save(os.path.join(flow_dir, f'flow_fwd_{i:05d}.npy'), flow_fwd_np)
            np.save(os.path.join(flow_dir, f'flow_bwd_{i:05d}.npy'), flow_bwd_np)
            flo_img = flow_viz.flow_to_image(flow_fwd_np)
            Image.fromarray(flo_img).save(os.path.join(flow_vis_dir, f'flow_{i:05d}.png'))
            if (i + 1) % 50 == 0:
                elapsed = time.time() - start
                fps = (i + 1) / elapsed
                eta = (len(images) - 1 - i - 1) / fps
                print(f"  [{i+1}/{len(images)-1}] {fps:.1f} pairs/s, ETA: {eta:.0f}s")
    elapsed = time.time() - start
    print(f"\nDone! {len(images)-1} flow pairs in {elapsed:.1f}s")

if __name__ == '__main__':
    main()
