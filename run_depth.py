"""Generate temporally consistent depth maps using Video Depth Anything (vits for 6GB VRAM)."""
import sys
sys.path.append(r'C:\Users\sean\src\upscale-experiment\Video-Depth-Anything')

import os
import torch
import numpy as np

from video_depth_anything.video_depth import VideoDepthAnything
from utils.dc_utils import read_video_frames, save_video

DEVICE = 'cuda'

def main():
    data_dir = r'C:\Users\sean\src\upscale-experiment\data'
    input_video = os.path.join(data_dir, 'clip_480p.mp4')
    output_dir = os.path.join(data_dir, 'depth_output')
    os.makedirs(output_dir, exist_ok=True)

    # Use vits (small) model to fit comfortably in 6GB VRAM
    model_config = {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
    checkpoint = r'C:\Users\sean\src\upscale-experiment\Video-Depth-Anything\checkpoints\video_depth_anything_vits.pth'

    print("Loading Video Depth Anything (vits)...")
    model = VideoDepthAnything(**model_config)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    model = model.to(DEVICE).eval()

    print(f"Reading video: {input_video}")
    frames, target_fps = read_video_frames(input_video, process_length=-1, max_res=480, target_fps=-1)
    print(f"  {len(frames)} frames at {target_fps} fps")

    print("Generating depth maps...")
    depths, fps = model.infer_video_depth(frames, target_fps, input_size=364, device=DEVICE)

    # Save depth video visualization
    save_video(depths, os.path.join(output_dir, 'depth_vis.mp4'), fps=fps, is_depths=True)
    save_video(frames, os.path.join(output_dir, 'depth_src.mp4'), fps=fps)

    # Save individual depth maps as numpy for the fusion pipeline
    depth_npy_dir = os.path.join(data_dir, 'depth_npy')
    os.makedirs(depth_npy_dir, exist_ok=True)
    for i, d in enumerate(depths):
        if isinstance(d, np.ndarray):
            np.save(os.path.join(depth_npy_dir, f'depth_{i:05d}.npy'), d)
        else:
            np.save(os.path.join(depth_npy_dir, f'depth_{i:05d}.npy'), np.array(d))

    print(f"\nDone! Depth maps saved to {output_dir} and {depth_npy_dir}")

if __name__ == '__main__':
    main()
