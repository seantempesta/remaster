import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.ffmpeg_utils import get_ffmpeg
import subprocess
ffmpeg = get_ffmpeg()
r = subprocess.run([ffmpeg, '-i', r'E:\plex\tv\Firefly (2002) Season 1 S01 (1080p BluRay x265 HEVC 10bit AAC Silence)\Firefly (2002) - S01E01 - Serenity (1080p x265 Silence).mkv'], capture_output=True, text=True)
for line in r.stderr.split('\n'):
    if any(k in line.lower() for k in ['stream #0:0', 'video:', 'duration:', 'bitrate:']):
        print(line.strip())
