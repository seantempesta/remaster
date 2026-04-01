import imageio_ffmpeg, subprocess
ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
r = subprocess.run([ffmpeg, '-i', r'E:\plex\tv\Firefly (2002) Season 1 S01 (1080p BluRay x265 HEVC 10bit AAC Silence)\Firefly (2002) - S01E01 - Serenity (1080p x265 Silence).mkv'], capture_output=True, text=True)
for line in r.stderr.split('\n'):
    if any(k in line.lower() for k in ['stream #0:0', 'video:', 'duration:', 'bitrate:']):
        print(line.strip())
