"""
Generic GPU runner for upscale-experiment.
Run any project script on a cloud GPU:

    modal run modal_runner.py --script denoise_scunet.py
    modal run modal_runner.py --script denoise_scunet.py --args "--some-flag value"
    modal run modal_runner.py --script bench_sdpa.py --gpu A100-40GB

Upload/download data:
    modal volume put upscale-data ./data/frames/ /frames/
    modal volume get upscale-data /results/ ./data/results/
    modal volume ls upscale-data /

Interactive debugging:
    modal shell modal_runner.py
"""
import modal
import subprocess
import sys

# Persistent storage shared between runs
vol = modal.Volume.from_name("upscale-data", create_if_missing=True)
VOL_MOUNT = "/mnt/data"

# Container environment matching our local conda env
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "torchaudio==2.5.1",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "realesrgan",
        "basicsr",
        "huggingface_hub",
        "scipy",
        "opencv-python-headless",
        "imageio-ffmpeg",
        "numpy",
        "pillow",
        "timm",
        "einops",
        "thop",
    )
    .add_local_dir(
        ".",
        remote_path="/root/project",
        ignore=[
            "data/**",
            ".git/**",
            "__pycache__/**",
            "*.pyc",
            "mmagic/**",
            "BasicVSR_PlusPlus/**",
        ],
    )
)

app = modal.App("upscale-runner", image=image)


@app.function(
    gpu="A10G",
    volumes={VOL_MOUNT: vol},
    timeout=7200,
)
def run_script(script_name: str, extra_args: str = ""):
    """Run any Python script from the project on a cloud GPU."""
    import shlex

    cmd = [sys.executable, script_name] + shlex.split(extra_args)
    print(f">>> Running: {' '.join(cmd)}")
    print(f">>> Volume mounted at: {VOL_MOUNT}")
    print(f">>> Working dir: /root/project")

    result = subprocess.run(
        cmd,
        cwd="/root/project",
    )

    vol.commit()

    if result.returncode != 0:
        raise SystemExit(f"Script exited with code {result.returncode}")

    print(">>> Done.")


@app.local_entrypoint()
def main(script: str, args: str = "", gpu: str = "A10G"):
    """
    Usage:
        modal run modal_runner.py --script bench_sdpa.py
        modal run modal_runner.py --script denoise_scunet.py --args "--input /mnt/data/frames"
    """
    print(f"Dispatching '{script}' to cloud GPU ({gpu})...")
    run_script.remote(script, args)
    print("Complete.")
