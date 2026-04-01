import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import imageio_ffmpeg, subprocess, os, glob
from lib.paths import DATA_DIR

ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
src = r'E:\plex\tv\Firefly (2002) Season 1 S01 (1080p BluRay x265 HEVC 10bit AAC Silence)\Firefly (2002) - S01E01 - Serenity (1080p x265 Silence).mkv'
outdir = str(DATA_DIR)
os.makedirs(outdir, exist_ok=True)

# Extract 30s clip at 1080p (ground truth)
print("Extracting 1080p ground truth clip...")
subprocess.run([ffmpeg, '-y', '-ss', '00:00:30', '-i', src, '-t', '30',
    '-c:v', 'libx264', '-crf', '14', '-an',
    os.path.join(outdir, 'clip_1080p.mp4')], check=True)

# Downscale to 480p (SD input for upscaling)
print("Creating 480p SD input...")
subprocess.run([ffmpeg, '-y', '-i', os.path.join(outdir, 'clip_1080p.mp4'),
    '-vf', 'scale=854:480', '-c:v', 'libx264', '-crf', '18', '-an',
    os.path.join(outdir, 'clip_480p.mp4')], check=True)

# Extract 480p frames for processing
frames_dir = os.path.join(outdir, 'frames_480p')
os.makedirs(frames_dir, exist_ok=True)
print("Extracting 480p frames...")
subprocess.run([ffmpeg, '-y', '-i', os.path.join(outdir, 'clip_480p.mp4'),
    os.path.join(frames_dir, 'frame_%05d.png')], check=True)

# Extract 1080p frames for ground truth comparison
gt_dir = os.path.join(outdir, 'frames_1080p_gt')
os.makedirs(gt_dir, exist_ok=True)
print("Extracting 1080p ground truth frames...")
subprocess.run([ffmpeg, '-y', '-i', os.path.join(outdir, 'clip_1080p.mp4'),
    os.path.join(gt_dir, 'frame_%05d.png')], check=True)

# Count frames
n480 = len(glob.glob(os.path.join(frames_dir, '*.png')))
n1080 = len(glob.glob(os.path.join(gt_dir, '*.png')))
print(f"\nDone! {n480} frames at 480p, {n1080} frames at 1080p ground truth")
