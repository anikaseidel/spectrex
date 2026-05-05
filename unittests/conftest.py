"""Shared pytest fixtures for spectrex unit tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="session")
def testdata_dir() -> Path:
    """Absolute path to the testdata directory at the repo root."""
    return Path(__file__).parent.parent / "testdata"


@pytest.fixture
def small_wavelengths() -> np.ndarray:
    """A tiny wavelength grid (Angstrom) for synthetic tests."""
    return np.linspace(8000.0, 18000.0, 12)


@pytest.fixture
def small_components(small_wavelengths) -> np.ndarray:
    """Synthetic (n_wav, 3) basis components."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((len(small_wavelengths), 3))
