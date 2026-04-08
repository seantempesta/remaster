"""Shared fixtures for inference quality tests."""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def project_root():
    return PROJECT_ROOT


def pytest_configure(config):
    """Add project root to sys.path so lib/ and models/ are importable."""
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
