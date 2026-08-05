"""Engine selection logic from remaster/vs_remaster.py.

A wrong engine shape or IO precision only surfaces as a VapourSynth error deep
into an encode, so pin the pure logic here. No GPU -- vapoursynth is stubbed,
since it only ships inside the portable tools/vs runtime.

Run: pytest tests/test_engine_selection.py -v
"""
import sys
import types
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENGINE_DIR = PROJECT_ROOT / "checkpoints" / "drunet_student"


@pytest.fixture(scope="module")
def vsr():
    """Import vs_remaster with a stubbed vapoursynth module."""
    stub = types.ModuleType("vapoursynth")
    stub.RGBH, stub.RGBS = "RGBH", "RGBS"
    stub.core = None
    sys.modules.setdefault("vapoursynth", stub)
    sys.path.insert(0, str(PROJECT_ROOT / "remaster"))
    import vs_remaster
    return vs_remaster


@pytest.mark.parametrize("name,expected", [
    ("drunet_student_1920x816_fp16.engine", (1920, 816)),
    ("drunet_student_1920x1080_fp16.engine", (1920, 1080)),
    ("drunet_student_1920x1080_int8.engine", (1920, 1080)),
    # Predates the {W}x{H} convention but is still the padding fallback.
    ("drunet_student_1080p_fp16.engine", (1920, 1080)),
])
def test_engine_shape(vsr, name, expected):
    assert vsr.engine_shape(name) == expected


def test_engine_shape_rejects_unparseable_name(vsr):
    with pytest.raises(ValueError, match="Cannot determine input shape"):
        vsr.engine_shape("mystery.engine")


@pytest.mark.parametrize("name,expected", [
    # fp16 IO needs 16-bit half input, everything else 32-bit float. Feeding
    # RGBS to an fp16 engine fails with "bits per sample mismatch".
    ("drunet_student_1920x816_fp16.engine", "RGBH"),
    ("drunet_student_1080p_int8.engine", "RGBS"),
    ("drunet_student_1080p_fp32.engine", "RGBS"),
])
def test_engine_format(vsr, name, expected):
    assert vsr.engine_format(name) == expected


def test_resolve_engine_prefers_native_resolution(vsr, monkeypatch, tmp_path):
    monkeypatch.delenv("REMASTER_ENGINE", raising=False)
    monkeypatch.setattr(vsr, "ENGINE_DIR", str(tmp_path))
    (tmp_path / "drunet_student_1920x816_fp16.engine").touch()

    assert Path(vsr.resolve_engine(1920, 816)).name == \
        "drunet_student_1920x816_fp16.engine"


def test_resolve_engine_finds_engine_rounded_up_to_multiple_of_8(vsr, monkeypatch,
                                                                 tmp_path):
    """A 1920x802 source is served by a 1920x808 engine.

    Engines can only be built on multiples of 8, so tools/build_trt_engine.py
    rounds up. If resolution here searched for the raw 802 it would never match
    what was built, rebuild forever, and quietly pad up to 1080p instead.
    """
    monkeypatch.delenv("REMASTER_ENGINE", raising=False)
    monkeypatch.setattr(vsr, "ENGINE_DIR", str(tmp_path))
    (tmp_path / "drunet_student_1920x808_fp16.engine").touch()

    assert Path(vsr.resolve_engine(1920, 802)).name == \
        "drunet_student_1920x808_fp16.engine"


def test_resolve_engine_rounding_matches_the_builder(vsr):
    """The two must agree, or built engines are never found again."""
    sys.path.insert(0, str(PROJECT_ROOT / "tools"))
    import build_trt_engine

    for w, h in [(1920, 802), (1920, 816), (1280, 715), (3840, 2160), (1920, 1080)]:
        assert (vsr.round_up(w), vsr.round_up(h)) == \
            (build_trt_engine.round_up(w), build_trt_engine.round_up(h))


def test_resolve_engine_falls_back_when_no_native_engine(vsr, monkeypatch, tmp_path):
    monkeypatch.delenv("REMASTER_ENGINE", raising=False)
    monkeypatch.setattr(vsr, "ENGINE_DIR", str(tmp_path))

    assert Path(vsr.resolve_engine(1920, 800)).name == vsr.LEGACY_1080P


def test_resolve_engine_env_override_wins(vsr, monkeypatch):
    monkeypatch.setenv("REMASTER_ENGINE", "X:/custom.engine")
    assert vsr.resolve_engine(1920, 816) == "X:/custom.engine"


def test_shipped_engines_follow_naming_convention(vsr):
    """Every engine on disk must be parseable, or enhance() dies mid-encode."""
    engines = list(ENGINE_DIR.glob("*.engine"))
    if not engines:
        pytest.skip("no engines built")
    for engine in engines:
        w, h = vsr.engine_shape(engine.name)
        assert w > 0 and h > 0
        assert h % 8 == 0 and w % 8 == 0, f"{engine.name} is not a multiple of 8"
