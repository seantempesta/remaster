"""Extract a 30-second clip from the middle of the episode — calmer scene."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import subprocess, os, glob
from lib.paths import DATA_DIR
from lib.ffmpeg_utils import get_ffmpeg

ffmpeg = get_ffmpeg()
src = r'E:\plex\tv\Firefly (2002) Season 1 S01 (1080p BluRay x265 HEVC 10bit AAC Silence)\Firefly (2002) - S01E01 - Serenity (1080p x265 Silence).mkv'
outdir = str(DATA_DIR)

# ~30 minutes in — should be a dialogue/interior scene on Serenity
print("Extracting 30s clip from 30:00...")
subprocess.run([ffmpeg, '-y', '-ss', '00:30:00', '-i', src, '-t', '30',
    '-c:v', 'libx264', '-crf', '14', '-an',
    os.path.join(outdir, 'clip_mid_1080p.mp4')], check=True)

# Extract frames
frames_dir = os.path.join(outdir, 'frames_mid_1080p')
os.makedirs(frames_dir, exist_ok=True)
print("Extracting frames...")
subprocess.run([ffmpeg, '-y', '-i', os.path.join(outdir, 'clip_mid_1080p.mp4'),
    os.path.join(frames_dir, 'frame_%05d.png')], check=True)

n = len(glob.glob(os.path.join(frames_dir, '*.png')))
print(f"Done! {n} frames extracted to {frames_dir}")
