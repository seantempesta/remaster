import vapoursynth as vs
import os, sys
core = vs.core

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
vsmlrt_path = os.path.join(project_root, "reference-code", "vs-mlrt", "scripts")
print(f"__file__: {__file__}", file=sys.stderr)
print(f"script_dir: {script_dir}", file=sys.stderr)
print(f"project_root: {project_root}", file=sys.stderr)
print(f"vsmlrt_path: {vsmlrt_path}", file=sys.stderr)
print(f"exists: {os.path.exists(vsmlrt_path)}", file=sys.stderr)
print(f"vsmlrt.py exists: {os.path.exists(os.path.join(vsmlrt_path, 'vsmlrt.py'))}", file=sys.stderr)
print(f"cwd: {os.getcwd()}", file=sys.stderr)

clip = core.std.BlankClip(width=16, height=16, length=1, format=vs.RGBS)
clip.set_output()
