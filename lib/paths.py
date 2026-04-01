"""Centralized path resolution for all project scripts.

Replaces hardcoded absolute paths with portable resolution that works
across local Windows, Modal containers, and different project locations.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REFERENCE_CODE = PROJECT_ROOT / "reference-code"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
BIN_DIR = PROJECT_ROOT / "bin"


def resolve_scunet_dir():
    """Search reference-code/SCUNet, legacy top-level SCUNet/, Modal /root/SCUNet."""
    candidates = [
        REFERENCE_CODE / "SCUNet",
        PROJECT_ROOT / "SCUNet",       # legacy fallback
        Path("/root/SCUNet"),          # Modal container
    ]
    for d in candidates:
        if (d / "models" / "network_scunet.py").exists():
            return d
    raise FileNotFoundError("SCUNet not found")


def add_scunet_to_path():
    """Add SCUNet to sys.path so `from models.network_scunet import SCUNet` works.
    Returns the SCUNet directory path."""
    d = resolve_scunet_dir()
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
    return d


def resolve_raft_dir():
    """Search reference-code/RAFT, legacy RAFT/, Modal /root/RAFT."""
    candidates = [
        REFERENCE_CODE / "RAFT",
        PROJECT_ROOT / "RAFT",
        Path("/root/RAFT"),
    ]
    for d in candidates:
        if (d / "core" / "raft.py").exists():
            return d
    raise FileNotFoundError("RAFT not found")


def add_raft_to_path():
    """Add RAFT/core to sys.path so `from raft import RAFT` works.
    Returns the RAFT root directory path."""
    d = resolve_raft_dir()
    core = d / "core"
    if str(core) not in sys.path:
        sys.path.insert(0, str(core))
    return d


def resolve_depth_dir():
    """Search reference-code/Video-Depth-Anything, legacy, Modal."""
    candidates = [
        REFERENCE_CODE / "Video-Depth-Anything",
        PROJECT_ROOT / "Video-Depth-Anything",
        Path("/root/Video-Depth-Anything"),
    ]
    for d in candidates:
        if (d / "video_depth_anything").is_dir():
            return d
    raise FileNotFoundError("Video-Depth-Anything not found")


def add_depth_to_path():
    """Add Video-Depth-Anything to sys.path.
    Returns the directory path."""
    d = resolve_depth_dir()
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
    return d
