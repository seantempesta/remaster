"""
Upload remaster DRUNet models to HuggingFace Hub.

Uploads student and teacher checkpoints, ONNX model, and model card to a
HuggingFace model repository. Authentication via HF_TOKEN environment variable.

Usage:
  # Set token via environment variable or .env file in project root
  export HF_TOKEN=hf_xxxxx
  # Or: echo "HF_TOKEN=hf_xxxxx" > .env

  # Upload all models (repo must already exist)
  python tools/upload_to_hf.py

  # Create repo and upload
  python tools/upload_to_hf.py --create-repo

  # Upload student only
  python tools/upload_to_hf.py --student-only

  # Custom repo name
  python tools/upload_to_hf.py --repo seantempesta/remaster-drunet-v2

  # Dry run (check files exist, don't upload)
  python tools/upload_to_hf.py --dry-run
"""
import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_token():
    """Get HuggingFace token from environment or .env file. Never hardcode tokens."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        # Try reading from .env file in project root
        env_path = os.path.join(PROJECT_ROOT, ".env")
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("HF_TOKEN=") and not line.startswith("#"):
                        token = line.split("=", 1)[1].strip().strip("'\"")
                        break
    if not token:
        print("ERROR: HF_TOKEN not found.")
        print("Set it via environment variable: export HF_TOKEN=hf_xxxxx")
        print("Or add HF_TOKEN=hf_xxxxx to .env file in project root.")
        print("Get a token at: https://huggingface.co/settings/tokens")
        sys.exit(1)
    return token


def resolve_files(student_only=False):
    """Resolve checkpoint files and verify they exist."""
    files = {}

    # Student model (always uploaded)
    student_pth = os.path.join(
        PROJECT_ROOT, "checkpoints", "drunet_student", "final.pth"
    )
    if os.path.exists(student_pth):
        files["drunet_student.pth"] = student_pth
    else:
        print(f"WARNING: Student checkpoint not found: {student_pth}")

    # Student ONNX
    student_onnx = os.path.join(
        PROJECT_ROOT, "checkpoints", "drunet_student", "drunet_student.onnx"
    )
    if os.path.exists(student_onnx):
        files["drunet_student.onnx"] = student_onnx
    else:
        print(f"WARNING: Student ONNX not found: {student_onnx}")

    # Teacher model (optional)
    if not student_only:
        teacher_pth = os.path.join(
            PROJECT_ROOT, "checkpoints", "drunet_teacher", "final.pth"
        )
        if os.path.exists(teacher_pth):
            files["drunet_teacher.pth"] = teacher_pth
        else:
            print(f"WARNING: Teacher checkpoint not found: {teacher_pth}")

    # Model card
    model_card = os.path.join(PROJECT_ROOT, "tools", "hf_model_card.md")
    if os.path.exists(model_card):
        files["README.md"] = model_card
    else:
        print(f"WARNING: Model card not found: {model_card}")

    return files


def upload(repo_id, files, token, create_pr=False, create_repo=False):
    """Upload files to HuggingFace Hub."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)

    if create_repo:
        print(f"Creating repo: {repo_id} (if not exists)")
        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            exist_ok=True,
        )
    else:
        print(f"Uploading to existing repo: {repo_id}")
        print("  (use --create-repo to create the repo if it doesn't exist)")

    # Upload each file
    for remote_name, local_path in files.items():
        size_mb = os.path.getsize(local_path) / 1024**2
        print(f"  Uploading {remote_name} ({size_mb:.1f} MB) ...")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=remote_name,
            repo_id=repo_id,
            repo_type="model",
            create_pr=create_pr,
        )
        print(f"  -> {remote_name} uploaded")

    url = f"https://huggingface.co/{repo_id}"
    print(f"\nDone! Model page: {url}")
    return url


def main():
    parser = argparse.ArgumentParser(
        description="Upload remaster DRUNet models to HuggingFace Hub"
    )
    parser.add_argument(
        "--repo",
        default="seantempesta/remaster-drunet",
        help="HuggingFace repo ID (default: seantempesta/remaster-drunet)",
    )
    parser.add_argument(
        "--student-only",
        action="store_true",
        help="Upload student model only (skip teacher)",
    )
    parser.add_argument(
        "--create-repo",
        action="store_true",
        help="Create the HuggingFace repo if it doesn't exist (off by default)",
    )
    parser.add_argument(
        "--create-pr",
        action="store_true",
        help="Create a pull request instead of committing directly",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check files and token, but don't upload",
    )
    args = parser.parse_args()

    # Validate token
    token = get_token()
    print(f"HF_TOKEN: {'*' * 8}...{token[-4:]}")

    # Resolve files
    files = resolve_files(student_only=args.student_only)

    if not files:
        print("ERROR: No files found to upload.")
        sys.exit(1)

    print(f"\nFiles to upload to {args.repo}:")
    for remote_name, local_path in files.items():
        size_mb = os.path.getsize(local_path) / 1024**2
        print(f"  {remote_name:30s} <- {local_path} ({size_mb:.1f} MB)")

    if args.dry_run:
        print("\nDry run -- no files uploaded.")
        # Validate token by checking auth
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        user = api.whoami()
        print(f"Authenticated as: {user['name']}")
        return

    upload(
        repo_id=args.repo,
        files=files,
        token=token,
        create_pr=args.create_pr,
        create_repo=args.create_repo,
    )


if __name__ == "__main__":
    main()
