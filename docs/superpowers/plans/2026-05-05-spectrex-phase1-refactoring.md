# specTrex Phase 1 Refactoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `legacy/Ex/` into a clean, tested `src/spectrex/` package with four focused modules, a `ForwardOperatorProtocol` seam for future JAX migration, and a comprehensive unit test suite.

**Architecture:** Four layers — `InstrumentConfig` (I/O, data loading), `EigenspectraBasis` (PCA spectral basis), `SciPySparseOperator` (forward model H), `SpectralSolver` (LSQR/LSMR inversion). A `ForwardOperatorProtocol` typed protocol decouples the solver from the operator implementation. All paths are explicit; no CWD assumptions; no hard-coded instrument strings.

**Tech Stack:** Python 3.12+, NumPy ≥ 2.0, SciPy, Astropy, grismagic, pytest, ruff, ty. uv-managed venv.

---

## File Map

| Action   | Path                                       | Responsibility                              |
|----------|--------------------------------------------|---------------------------------------------|
| Create   | `src/spectrex/__init__.py`                 | Public re-exports                           |
| Create   | `src/spectrex/instrument.py`               | `InstrumentConfig` — data loading only      |
| Create   | `src/spectrex/basis.py`                    | `EigenspectraBasis` — PCA basis             |
| Create   | `src/spectrex/operator.py`                 | `ForwardOperatorProtocol` + `SciPySparseOperator` |
| Create   | `src/spectrex/solver.py`                   | `NoiseModel` + `SpectralSolver`             |
| Create   | `testdata/`                                | Non-shipped test fixtures (copied from `legacy/Ex/`) |
| Create   | `unittests/conftest.py`                    | Shared pytest fixtures                      |
| Create   | `unittests/test_basis.py`                  | Basis unit tests                            |
| Create   | `unittests/test_instrument.py`             | Instrument unit tests                       |
| Create   | `unittests/test_operator.py`               | Operator unit tests                         |
| Create   | `unittests/test_solver.py`                 | Solver unit tests                           |
| Create   | `unittests/test_integration.py`            | Slow round-trip integration tests           |
| Modify   | `pyproject.toml`                           | Add pytest config, markers, testpaths       |
| Modify   | `.gitignore`                               | Exclude `.agents-workspace/`                |

---

## Task 0: Infrastructure

**Files:**
- Modify: `pyproject.toml`
- Modify: `.gitignore`
- Create: `testdata/` (copy from `legacy/Ex/`)
- Create: `unittests/conftest.py`

- [ ] **Step 1: Copy testdata files from legacy**

```bash
mkdir -p testdata/SenseConfig
cp -r "legacy/Ex/Config Files" testdata/
cp -r "legacy/Ex/SenseConfig/wfss-grism-configuration" testdata/SenseConfig/
cp legacy/Ex/jwst_niriss_wavelengthrange_0002.asdf testdata/
cp legacy/Ex/eigenspectra_kurucz.csv testdata/
```

- [ ] **Step 2: Add pytest configuration to `pyproject.toml`**

Add this block at the end of `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = ["slow: marks tests as slow (deselect with '-m not slow')"]
testpaths = ["unittests"]
```

- [ ] **Step 3: Exclude `.agents-workspace/` in `.gitignore`**

Append to `.gitignore`:

```
.agents-workspace/
```

- [ ] **Step 4: Write `unittests/conftest.py`**

```python
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
```

- [ ] **Step 5: Verify pytest collects without errors**

```bash
uv run pytest --collect-only
```

Expected: `0 errors`, collection completes (zero tests is fine at this point).

- [ ] **Step 6: Commit**

```bash
git add testdata/ unittests/conftest.py pyproject.toml .gitignore
git commit -m "chore: add testdata, pytest config, conftest skeleton"
```

---

## Task 1: `EigenspectraBasis` (basis.py)

**Files:**
- Create: `src/spectrex/basis.py`
- Create: `unittests/test_basis.py`

- [ ] **Step 1: Write the failing tests in `unittests/test_basis.py`**

```python
"""Unit tests for spectrex.basis.EigenspectraBasis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spectrex.basis import EigenspectraBasis


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_csv(tmp_path: Path) -> Path:
    """A minimal eigenspectra CSV: 20 wavelength points (µm), 3 components."""
    rng = np.random.default_rng(0)
    wav_um = np.linspace(0.7, 2.2, 20)
    components = rng.standard_normal((20, 3))
    data = np.column_stack([wav_um, components])
    path = tmp_path / "eigenspectra.csv"
    np.savetxt(path, data, delimiter=",",
               header="wavelength,c0,c1,c2", comments="")
    return path


@pytest.fixture
def target_wavelengths() -> np.ndarray:
    return np.linspace(8000.0, 18000.0, 15)  # Angstrom, within [7000, 22000]


@pytest.fixture
def basis(synthetic_csv, target_wavelengths) -> EigenspectraBasis:
    return EigenspectraBasis.from_csv(synthetic_csv, target_wavelengths)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_from_csv_shapes(basis, target_wavelengths):
    assert basis.wavelengths.shape == (15,)
    assert basis.components.shape == (15, 3)
    assert basis.n_components == 3


def test_from_csv_wavelengths_match(basis, target_wavelengths):
    np.testing.assert_array_equal(basis.wavelengths, target_wavelengths)


def test_from_csv_out_of_range_raises(synthetic_csv):
    """Request wavelengths outside the CSV range should raise ValueError."""
    too_wide = np.linspace(4000.0, 30000.0, 15)
    with pytest.raises(ValueError, match="wavelengths"):
        EigenspectraBasis.from_csv(synthetic_csv, too_wide)


def test_components_read_only(basis):
    with pytest.raises((ValueError, TypeError)):
        basis.components[0, 0] = 999.0


# ---------------------------------------------------------------------------
# reconstruct
# ---------------------------------------------------------------------------

def test_reconstruct_shape(basis):
    coeffs = np.ones(3)
    spectrum = basis.reconstruct(coeffs)
    assert spectrum.shape == (15,)


def test_reconstruct_first_component(basis):
    """reconstruct([1,0,0]) should return the first component column."""
    coeffs = np.array([1.0, 0.0, 0.0])
    result = basis.reconstruct(coeffs)
    np.testing.assert_allclose(result, basis.components[:, 0])


def test_reconstruct_linear(basis):
    """reconstruct(a + b) == reconstruct(a) + reconstruct(b)."""
    rng = np.random.default_rng(1)
    a = rng.standard_normal(3)
    b = rng.standard_normal(3)
    np.testing.assert_allclose(
        basis.reconstruct(a + b),
        basis.reconstruct(a) + basis.reconstruct(b),
    )


# ---------------------------------------------------------------------------
# integrated_weights
# ---------------------------------------------------------------------------

def test_integrated_weights_shape(basis):
    w = basis.integrated_weights()
    assert w.shape == (3,)


def test_integrated_weights_matches_trapezoid(basis):
    expected = np.trapezoid(basis.components, basis.wavelengths, axis=0)
    np.testing.assert_allclose(basis.integrated_weights(), expected)


# ---------------------------------------------------------------------------
# broadband_image
# ---------------------------------------------------------------------------

def test_broadband_image_shape(basis):
    n_rows, n_cols = 4, 5
    a_tilde = np.zeros(n_rows * n_cols * 3)
    img = basis.broadband_image(a_tilde, (n_rows, n_cols))
    assert img.shape == (n_rows, n_cols)


def test_broadband_image_zeros(basis):
    a_tilde = np.zeros(4 * 5 * 3)
    img = basis.broadband_image(a_tilde, (4, 5))
    np.testing.assert_array_equal(img, 0.0)


def test_broadband_image_matches_loop(basis):
    """Vectorised result must match a naive pixel-by-pixel loop."""
    rng = np.random.default_rng(7)
    n_rows, n_cols = 3, 4
    a_tilde = rng.standard_normal(n_rows * n_cols * 3)
    w = basis.integrated_weights()

    expected = np.zeros((n_rows, n_cols))
    for i in range(n_rows):
        for j in range(n_cols):
            k = i * n_cols + j
            expected[i, j] = a_tilde[k * 3 : (k + 1) * 3] @ w

    result = basis.broadband_image(a_tilde, (n_rows, n_cols))
    np.testing.assert_allclose(result, expected, rtol=1e-12)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest unittests/test_basis.py -v 2>&1 | head -20
```

Expected: `ImportError: cannot import name 'EigenspectraBasis' from 'spectrex.basis'` (module doesn't exist yet).

- [ ] **Step 3: Create `src/spectrex/basis.py`**

```python
"""PCA eigenspectra basis for spectral decomposition."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EigenspectraBasis:
    """PCA eigenspectra basis for representing source spectra.

    Parameters
    ----------
    wavelengths : np.ndarray
        Wavelength grid in Angstrom, shape ``(n_wav,)``. Read-only.
    components : np.ndarray
        Basis components, shape ``(n_wav, n_components)``. Read-only.
    n_components : int
        Number of basis components (``components.shape[1]``).
    """

    wavelengths: np.ndarray
    components: np.ndarray
    n_components: int

    @classmethod
    def from_csv(
        cls,
        csv_path: Path,
        wavelengths: np.ndarray,
    ) -> "EigenspectraBasis":
        """Load and interpolate eigenspectra from a CSV file.

        Parameters
        ----------
        csv_path : Path
            CSV file with a header row, first column wavelength in µm,
            remaining columns as eigenspectra components.
        wavelengths : np.ndarray
            Target wavelength grid in Angstrom to interpolate onto.

        Returns
        -------
        EigenspectraBasis

        Raises
        ------
        ValueError
            If ``wavelengths`` lies outside the CSV wavelength range.
        """
        data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
        wav_angstrom = data[:, 0] * 1e4  # µm -> Angstrom
        components_raw = data[:, 1:]

        lo, hi = wav_angstrom.min(), wav_angstrom.max()
        if wavelengths.min() < lo or wavelengths.max() > hi:
            raise ValueError(
                f"Requested wavelengths [{wavelengths.min():.0f}, "
                f"{wavelengths.max():.0f}] Å lie outside CSV range "
                f"[{lo:.0f}, {hi:.0f}] Å."
            )

        n_components = components_raw.shape[1]
        interpolated = np.column_stack([
            np.interp(wavelengths, wav_angstrom, components_raw[:, m])
            for m in range(n_components)
        ])
        interpolated.setflags(write=False)

        wav_copy = wavelengths.copy()
        wav_copy.setflags(write=False)

        logger.debug(
            "Loaded %d eigenspectra components from %s", n_components, csv_path
        )
        return cls(
            wavelengths=wav_copy,
            components=interpolated,
            n_components=n_components,
        )

    def reconstruct(self, coefficients: np.ndarray) -> np.ndarray:
        """Reconstruct a spectrum from PCA coefficients.

        Parameters
        ----------
        coefficients : np.ndarray
            Shape ``(n_components,)``.

        Returns
        -------
        np.ndarray
            Shape ``(n_wav,)`` — flux at each wavelength.
        """
        return self.components @ coefficients

    def integrated_weights(self) -> np.ndarray:
        """Trapezoidal integral of each basis component over wavelength.

        Computed as ``np.trapezoid(components, wavelengths, axis=0)``.

        Returns
        -------
        np.ndarray
            Shape ``(n_components,)``. Dot with a pixel's coefficients
            gives its broadband flux.
        """
        return np.trapezoid(self.components, self.wavelengths, axis=0)

    def broadband_image(
        self,
        a_tilde: np.ndarray,
        image_shape: tuple[int, int],
    ) -> np.ndarray:
        """Reconstruct broadband direct image from full coefficient vector.

        Vectorised; no Python loop over pixels.

        Parameters
        ----------
        a_tilde : np.ndarray
            Shape ``(n_rows * n_cols * n_components,)``.
        image_shape : tuple[int, int]
            ``(n_rows, n_cols)``.

        Returns
        -------
        np.ndarray
            Shape ``(n_rows, n_cols)``.
        """
        n_rows, n_cols = image_shape
        n_pix = n_rows * n_cols
        w = self.integrated_weights()                     # (n_components,)
        a = a_tilde.reshape(n_pix, self.n_components)    # (n_pix, n_components)
        return (a @ w).reshape(n_rows, n_cols)            # (n_rows, n_cols)
```

- [ ] **Step 4: Create a minimal `src/spectrex/__init__.py`** (will be expanded later)

```python
"""specTrex — Grism spectra extraction in crowded regions."""

from spectrex._version import version as __version__

__all__ = ["__version__"]
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest unittests/test_basis.py -v
```

Expected: all 12 tests PASS.

- [ ] **Step 6: Run ruff and ty**

```bash
uv run ruff check src/spectrex/basis.py
uv run ty check src/spectrex/basis.py
```

Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add src/spectrex/basis.py src/spectrex/__init__.py unittests/test_basis.py
git commit -m "feat: add EigenspectraBasis with from_csv, reconstruct, broadband_image"
```

---

## Task 2: `InstrumentConfig` (instrument.py)

**Files:**
- Create: `src/spectrex/instrument.py`
- Create: `unittests/test_instrument.py`

- [ ] **Step 1: Write the failing tests in `unittests/test_instrument.py`**

```python
"""Unit tests for spectrex.instrument.InstrumentConfig."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from astropy.io import fits

from spectrex.instrument import InstrumentConfig


# ---------------------------------------------------------------------------
# Helpers: build a fake sensitivity directory with FITS files
# ---------------------------------------------------------------------------

def make_sensitivity_dir(tmp_path: Path) -> Path:
    """Write minimal sensitivity FITS files for orders 0, 1, 2."""
    sens_dir = tmp_path / "SenseConfig" / "wfss-grism-configuration"
    sens_dir.mkdir(parents=True)
    wavelengths = np.linspace(8000.0, 18000.0, 50)  # Angstrom
    sensitivity = np.ones(50, dtype=float)
    for order_int in [0, 1, 2]:
        col1 = fits.Column(name="WAVELENGTH", format="D", array=wavelengths)
        col2 = fits.Column(name="SENSITIVITY", format="D", array=sensitivity)
        hdu = fits.BinTableHDU.from_columns([col1, col2])
        fname = f"NIRISS.GR150R.F150W.{order_int}.etc.1.5.2.sens.fits"
        hdu.writeto(sens_dir / fname)
    return sens_dir


def make_mock_trace() -> MagicMock:
    """Return a GrismTrace mock that reports orders A, B, C and lam_range."""
    trace = MagicMock()
    trace.orders = ["A", "B", "C"]
    trace._lam_range.return_value = (0.8, 1.7)  # µm

    def _get_trace(x0, y0, order, lam):
        return np.full_like(lam, x0 + 1.0), np.full_like(lam, y0)

    trace.get_trace_at_wavelength.side_effect = _get_trace
    return trace


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sens_dir(tmp_path) -> Path:
    return make_sensitivity_dir(tmp_path)


@pytest.fixture
def mock_trace() -> MagicMock:
    return make_mock_trace()


@pytest.fixture
def config(mock_trace, sens_dir) -> InstrumentConfig:
    with patch("spectrex.instrument.GrismTrace") as MockGT:
        MockGT.from_file.return_value = mock_trace
        return InstrumentConfig.from_files(
            conf_path=Path("GR150R.F150W.220725.conf"),
            wavelengthrange_path=Path("dummy.asdf"),
            sensitivity_dir=sens_dir,
            filter_name="F150W",
            n_wavelengths=20,
        )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_wavelengths_shape(config):
    assert config.wavelengths.shape == (20,)


def test_wavelengths_clipped(config):
    assert config.wavelengths.min() >= 7000.0
    assert config.wavelengths.max() <= 22000.0


def test_wavelengths_within_lam_range(config):
    """Wavelength grid must lie within [lo, hi] derived from the mock trace."""
    # mock returns (0.8, 1.7) µm = (8000, 17000) Å clipped to [7000, 22000]
    assert config.wavelengths.min() >= 8000.0
    assert config.wavelengths.max() <= 17000.0


def test_orders(config):
    assert config.orders == ["A", "B", "C"]


def test_grism_inferred(config):
    assert config.grism == "GR150R"


def test_filter_name(config):
    assert config.filter_name == "F150W"


# ---------------------------------------------------------------------------
# Sensitivity curves
# ---------------------------------------------------------------------------

def test_sensitivity_keys(config):
    for order in ["A", "B", "C"]:
        assert order in config.sensitivity


def test_sensitivity_shape(config):
    for order in ["A", "B", "C"]:
        assert config.sensitivity[order].shape == (20,)


def test_sensitivity_nonneg(config):
    for order in ["A", "B", "C"]:
        assert np.all(config.sensitivity[order] >= 0.0)


def test_sensitivity_nonzero(config):
    for order in ["A", "B", "C"]:
        assert np.sum(config.sensitivity[order]) > 0.0


# ---------------------------------------------------------------------------
# get_trace
# ---------------------------------------------------------------------------

def test_get_trace_shapes(config):
    x_trace, y_trace = config.get_trace(5.0, 3.0, order="A")
    assert x_trace.shape == config.wavelengths.shape
    assert y_trace.shape == config.wavelengths.shape


def test_get_trace_bad_order_raises(config):
    with pytest.raises(ValueError, match="order"):
        config.get_trace(0.0, 0.0, order="Z")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest unittests/test_instrument.py -v 2>&1 | head -20
```

Expected: `ImportError: cannot import name 'InstrumentConfig' from 'spectrex.instrument'`.

- [ ] **Step 3: Create `src/spectrex/instrument.py`**

```python
"""Instrument configuration for JWST NIRISS grism spectroscopy."""

from __future__ import annotations

import glob
import logging
from pathlib import Path

import numpy as np
from astropy.io import fits
from grismagic.traces import GrismTrace

logger = logging.getLogger(__name__)

# PCA basis wavelength limits in Angstrom
_WAV_MIN_ANGSTROM: float = 7000.0
_WAV_MAX_ANGSTROM: float = 22000.0

# GrismTrace order letter -> sensitivity file order integer
_ORDER_LETTER_TO_INT: dict[str, int] = {"A": 1, "B": 0, "C": 2}


class InstrumentConfig:
    """Instrument configuration for a specific grism/filter combination.

    All instrument data is loaded eagerly at construction via
    :meth:`from_files`. No file I/O occurs after that point.

    Parameters
    ----------
    grism : str
        Grism identifier, e.g. ``"GR150R"``.
    filter_name : str
        Filter identifier, e.g. ``"F150W"``.
    wavelengths : np.ndarray
        Shared wavelength grid in Angstrom, shape ``(n_wav,)``.
    orders : list[str]
        Diffraction order labels, e.g. ``["A", "B", "C"]``.
    sensitivity : dict[str, np.ndarray]
        Per-order sensitivity curves, each shape ``(n_wav,)``,
        normalised so that ``sum(sensitivity) == 1`` (approximately).
    """

    def __init__(
        self,
        grism: str,
        filter_name: str,
        wavelengths: np.ndarray,
        orders: list[str],
        sensitivity: dict[str, np.ndarray],
        trace: GrismTrace,
    ) -> None:
        self.grism = grism
        self.filter_name = filter_name
        self.wavelengths = wavelengths
        self.orders = orders
        self.sensitivity = sensitivity
        self._trace = trace

    @classmethod
    def from_files(
        cls,
        conf_path: Path,
        wavelengthrange_path: Path,
        sensitivity_dir: Path,
        filter_name: str,
        n_wavelengths: int = 150,
    ) -> "InstrumentConfig":
        """Build an ``InstrumentConfig`` from calibration files.

        Parameters
        ----------
        conf_path : Path
            Path to the grism ``.conf`` configuration file.
            The grism name is inferred from the filename stem
            (e.g. ``GR150R.F150W.220725.conf`` -> ``"GR150R"``).
        wavelengthrange_path : Path
            Path to the JWST NIRISS wavelength-range ``.asdf`` file.
        sensitivity_dir : Path
            Directory containing sensitivity ``.fits`` files named
            ``NIRISS.{grism}.{filter_name}.{order_int}.*.sens.fits``.
        filter_name : str
            Filter name, e.g. ``"F150W"``.
        n_wavelengths : int, optional
            Number of wavelength sampling points. Default 150.

        Returns
        -------
        InstrumentConfig
        """
        trace = GrismTrace.from_file(
            conf_path, filter_name, wavelengthrange_path
        )

        lo_um, hi_um = trace._lam_range("1", None, None)
        lo = max(float(lo_um) * 1e4, _WAV_MIN_ANGSTROM)
        hi = min(float(hi_um) * 1e4, _WAV_MAX_ANGSTROM)
        wavelengths = np.linspace(lo, hi, n_wavelengths)

        grism = Path(conf_path).stem.split(".")[0]
        orders = list(trace.orders)

        sensitivity: dict[str, np.ndarray] = {}
        for order_letter in orders:
            order_int = _ORDER_LETTER_TO_INT.get(order_letter)
            if order_int is None:
                logger.debug(
                    "Unknown order letter %s; no sensitivity loaded.", order_letter
                )
                continue
            pattern = str(
                Path(sensitivity_dir)
                / f"NIRISS.{grism}.{filter_name}.{order_int}.*.sens.fits"
            )
            matches = glob.glob(pattern)
            if not matches:
                logger.warning(
                    "No sensitivity file found for order %s (pattern: %s).",
                    order_letter,
                    pattern,
                )
                continue
            with fits.open(matches[0]) as hdul:
                data = hdul[1].data
                wav = np.asarray(data["WAVELENGTH"], dtype=float)
                sens = np.asarray(data["SENSITIVITY"], dtype=float)
            total = sens.sum()
            if total > 0.0:
                sens = sens / total
            sensitivity[order_letter] = np.interp(
                wavelengths, wav, sens, left=0.0, right=0.0
            )
            logger.debug(
                "Loaded sensitivity for order %s from %s.",
                order_letter,
                matches[0],
            )

        logger.debug(
            "InstrumentConfig: grism=%s filter=%s "
            "wavelengths=[%.0f, %.0f] Å orders=%s",
            grism,
            filter_name,
            wavelengths[0],
            wavelengths[-1],
            orders,
        )
        return cls(
            grism=grism,
            filter_name=filter_name,
            wavelengths=wavelengths,
            orders=orders,
            sensitivity=sensitivity,
            trace=trace,
        )

    def get_trace(
        self,
        x0: float,
        y0: float,
        order: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return trace pixel coordinates for a source at ``(x0, y0)``.

        Parameters
        ----------
        x0 : float
            Source row position in detector coordinates.
        y0 : float
            Source column position in detector coordinates.
        order : str
            Diffraction order label, e.g. ``"A"``.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(x_trace, y_trace)`` arrays at ``self.wavelengths``,
            each shape ``(n_wav,)``.

        Raises
        ------
        ValueError
            If ``order`` is not in ``self.orders``.
        """
        if order not in self.orders:
            raise ValueError(
                f"order '{order}' is not among configured orders {self.orders}."
            )
        x_trace, y_trace = self._trace.get_trace_at_wavelength(
            x0, y0, order=order, lam=self.wavelengths
        )
        return np.asarray(x_trace), np.asarray(y_trace)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest unittests/test_instrument.py -v
```

Expected: all 14 tests PASS.

- [ ] **Step 5: Run ruff and ty**

```bash
uv run ruff check src/spectrex/instrument.py
uv run ty check src/spectrex/instrument.py
```

Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/spectrex/instrument.py unittests/test_instrument.py
git commit -m "feat: add InstrumentConfig with from_files, get_trace, sensitivity loading"
```

---

## Task 3: `ForwardOperatorProtocol` and `SciPySparseOperator` (operator.py)

**Files:**
- Create: `src/spectrex/operator.py`
- Create: `unittests/test_operator.py`

- [ ] **Step 1: Write the failing tests in `unittests/test_operator.py`**

```python
"""Unit tests for spectrex.operator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from spectrex.operator import ForwardOperatorProtocol, SciPySparseOperator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_H() -> csr_matrix:
    """A random dense 12x24 matrix converted to CSR (image 4x3, 2 components)."""
    rng = np.random.default_rng(1)
    return csr_matrix(rng.standard_normal((12, 24)))


@pytest.fixture
def small_operator(small_H) -> SciPySparseOperator:
    return SciPySparseOperator(small_H, image_shape=(4, 3))


# ---------------------------------------------------------------------------
# Attributes
# ---------------------------------------------------------------------------

def test_image_shape(small_operator):
    assert small_operator.image_shape == (4, 3)


def test_n_coefficients(small_operator):
    assert small_operator.n_coefficients == 24


# ---------------------------------------------------------------------------
# apply / apply_adjoint shapes
# ---------------------------------------------------------------------------

def test_apply_shape(small_operator):
    a = np.ones(24)
    result = small_operator.apply(a)
    assert result.shape == (12,)


def test_apply_adjoint_shape(small_operator):
    f = np.ones(12)
    result = small_operator.apply_adjoint(f)
    assert result.shape == (24,)


# ---------------------------------------------------------------------------
# Adjoint property
# ---------------------------------------------------------------------------

def test_adjoint_property(small_operator):
    """<v, H u> == <H^T v, u> for random u, v."""
    rng = np.random.default_rng(99)
    u = rng.standard_normal(24)
    v = rng.standard_normal(12)
    lhs = v @ small_operator.apply(u)
    rhs = u @ small_operator.apply_adjoint(v)
    np.testing.assert_allclose(lhs, rhs, rtol=1e-12)


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------

def test_save_load_roundtrip(small_operator, tmp_path):
    path = tmp_path / "test_op.npz"
    small_operator.save(path)
    loaded = SciPySparseOperator.load(path)

    assert loaded.image_shape == small_operator.image_shape
    assert loaded.n_coefficients == small_operator.n_coefficients

    a = np.ones(24)
    np.testing.assert_allclose(loaded.apply(a), small_operator.apply(a))


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------

def test_protocol_isinstance(small_operator):
    assert isinstance(small_operator, ForwardOperatorProtocol)


def test_protocol_attributes_present(small_operator):
    assert hasattr(small_operator, "image_shape")
    assert hasattr(small_operator, "n_coefficients")
    assert callable(small_operator.apply)
    assert callable(small_operator.apply_adjoint)


# ---------------------------------------------------------------------------
# build() with a synthetic tiny config — no real files needed
# ---------------------------------------------------------------------------

def test_build_shape(tmp_path):
    """build() with a mock config/basis produces an operator with correct shapes."""
    from unittest.mock import MagicMock

    from spectrex.basis import EigenspectraBasis

    n_wav = 8
    n_comp = 2
    image_shape = (5, 4)
    n_pix = 5 * 4

    # Synthetic basis
    rng = np.random.default_rng(3)
    wav = np.linspace(8000.0, 16000.0, n_wav)
    comps = np.abs(rng.standard_normal((n_wav, n_comp)))  # positive for realism
    comps.setflags(write=False)
    wav_ro = wav.copy(); wav_ro.setflags(write=False)
    basis = EigenspectraBasis(wavelengths=wav_ro, components=comps, n_components=n_comp)

    # Mock config: traces go to pixel (x0+1, y0) for all wavelengths
    config = MagicMock()
    config.orders = ["A"]
    config.sensitivity = {"A": np.ones(n_wav)}
    config.wavelengths = wav

    def _fake_trace(x0, y0, order, lam=None):
        # Clamp to image bounds so all traces land inside
        x_t = np.clip(np.full(n_wav, x0), 0, image_shape[0] - 1).astype(float)
        y_t = np.clip(np.full(n_wav, y0), 0, image_shape[1] - 1).astype(float)
        return x_t, y_t

    config.get_trace.side_effect = _fake_trace

    op = SciPySparseOperator.build(config, basis, image_shape)

    assert op.image_shape == image_shape
    assert op.n_coefficients == n_pix * n_comp
    # H shape: (n_pix, n_pix * n_comp)
    assert op._H.shape == (n_pix, n_pix * n_comp)


def test_build_apply_shape(tmp_path):
    """Operator built from scratch: apply() returns correct shape."""
    from unittest.mock import MagicMock

    from spectrex.basis import EigenspectraBasis

    n_wav, n_comp = 6, 2
    image_shape = (4, 3)
    n_pix = 4 * 3

    rng = np.random.default_rng(5)
    wav = np.linspace(8000.0, 16000.0, n_wav)
    comps = np.abs(rng.standard_normal((n_wav, n_comp)))
    comps.setflags(write=False)
    wav_ro = wav.copy(); wav_ro.setflags(write=False)
    basis = EigenspectraBasis(wavelengths=wav_ro, components=comps, n_components=n_comp)

    config = MagicMock()
    config.orders = ["A"]
    config.sensitivity = {"A": np.ones(n_wav)}
    config.wavelengths = wav

    def _fake_trace(x0, y0, order, lam=None):
        x_t = np.clip(np.full(n_wav, x0), 0, image_shape[0] - 1).astype(float)
        y_t = np.clip(np.full(n_wav, y0), 0, image_shape[1] - 1).astype(float)
        return x_t, y_t

    config.get_trace.side_effect = _fake_trace

    op = SciPySparseOperator.build(config, basis, image_shape)
    a_tilde = rng.standard_normal(op.n_coefficients)
    result = op.apply(a_tilde)
    assert result.shape == (n_pix,)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest unittests/test_operator.py -v 2>&1 | head -20
```

Expected: `ImportError: cannot import name 'SciPySparseOperator' from 'spectrex.operator'`.

- [ ] **Step 3: Create `src/spectrex/operator.py`**

```python
"""Grism forward operator: protocol and scipy sparse implementation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.sparse import csr_matrix, load_npz, save_npz

from spectrex.basis import EigenspectraBasis
from spectrex.instrument import InstrumentConfig

logger = logging.getLogger(__name__)


@runtime_checkable
class ForwardOperatorProtocol(Protocol):
    """Protocol for the grism dispersion operator H.

    Any object satisfying this protocol can be passed to
    :class:`~spectrex.solver.SpectralSolver`. The Phase 1 implementation
    is :class:`SciPySparseOperator`; Phase 2 will provide a JAX-based
    implementation without materialising H.

    Attributes
    ----------
    image_shape : tuple[int, int]
        ``(n_rows, n_cols)`` of the detector image.
    n_coefficients : int
        Total length of the ``a_tilde`` coefficient vector,
        equal to ``n_rows * n_cols * n_components``.
    """

    image_shape: tuple[int, int]
    n_coefficients: int
    # n_coefficients == image_shape[0] * image_shape[1] * basis.n_components

    def apply(self, a_tilde: np.ndarray) -> np.ndarray:
        """Forward pass: ``H @ a_tilde``.

        Parameters
        ----------
        a_tilde : np.ndarray
            Coefficient vector, shape ``(n_coefficients,)``.

        Returns
        -------
        np.ndarray
            Flattened dispersed image, shape ``(n_rows * n_cols,)``.
        """
        ...

    def apply_adjoint(self, f: np.ndarray) -> np.ndarray:
        """Adjoint pass: ``H.T @ f``.

        Parameters
        ----------
        f : np.ndarray
            Flattened dispersed image, shape ``(n_rows * n_cols,)``.

        Returns
        -------
        np.ndarray
            Coefficient vector, shape ``(n_coefficients,)``.
        """
        ...


class SciPySparseOperator:
    """Grism forward operator backed by a scipy CSR sparse matrix.

    Build from calibration data with :meth:`build`, or load a previously
    cached operator with :meth:`load`.

    Parameters
    ----------
    H : csr_matrix
        Sparse forward matrix, shape
        ``(n_rows * n_cols, n_rows * n_cols * n_components)``.
    image_shape : tuple[int, int]
        ``(n_rows, n_cols)`` of the detector image.
    """

    def __init__(
        self,
        H: csr_matrix,
        image_shape: tuple[int, int],
    ) -> None:
        self._H = H
        self.image_shape = image_shape
        self.n_coefficients: int = H.shape[1]

    @classmethod
    def build(
        cls,
        config: InstrumentConfig,
        basis: EigenspectraBasis,
        image_shape: tuple[int, int],
    ) -> "SciPySparseOperator":
        """Build the sparse forward matrix H from scratch.

        Parameters
        ----------
        config : InstrumentConfig
        basis : EigenspectraBasis
        image_shape : tuple[int, int]
            ``(n_rows, n_cols)`` of the detector image.

        Returns
        -------
        SciPySparseOperator

        Notes
        -----
        Build complexity is ``O(n_rows * n_cols * n_wavelengths)`` per
        diffraction order. For full NIRISS (2048 × 2048) this will take
        minutes. Cache the result with :meth:`save`.

        This method is marked ``@pytest.mark.slow`` in integration tests.
        """
        n_rows, n_cols = image_shape
        n_pix = n_rows * n_cols
        h = basis.n_components
        Phi_base = basis.components  # (n_wav, h)

        row_idx: list[np.ndarray] = []
        col_idx: list[np.ndarray] = []
        data_list: list[np.ndarray] = []

        for order in config.orders:
            sens = config.sensitivity.get(order)
            if sens is None:
                logger.debug("No sensitivity for order %s; skipping.", order)
                continue

            # Scale basis by sensitivity once per order: (n_wav, h)
            Phi = Phi_base * sens[:, np.newaxis]

            for i in range(n_rows):
                for j in range(n_cols):
                    k = i * n_cols + j  # source pixel flat index

                    try:
                        x_trace, y_trace = config.get_trace(
                            float(i), float(j), order=order
                        )
                    except (ValueError, IndexError) as exc:
                        logger.debug(
                            "get_trace failed at (%d, %d) order %s: %s",
                            i, j, order, exc,
                        )
                        continue

                    x_pix = np.round(x_trace).astype(int)
                    y_pix = np.round(y_trace).astype(int)

                    mask = (
                        (x_pix >= 0) & (x_pix < n_rows)
                        & (y_pix >= 0) & (y_pix < n_cols)
                    )
                    if not np.any(mask):
                        continue

                    x_valid = x_pix[mask]
                    y_valid = y_pix[mask]
                    lam_idx = np.where(mask)[0]

                    # Row indices in H for the dispersed pixels
                    rows_h = x_valid * n_cols + y_valid   # (n_valid,)
                    # Phi values at valid wavelengths: (n_valid, h)
                    phi_valid = Phi[lam_idx, :]

                    # Vectorise over basis components — no inner Python loop
                    cols_m = np.arange(k * h, (k + 1) * h)       # (h,)
                    n_valid = len(rows_h)
                    rows_block = np.repeat(rows_h, h)              # (n_valid*h,)
                    cols_block = np.tile(cols_m, n_valid)          # (n_valid*h,)
                    data_block = phi_valid.ravel(order="C")        # (n_valid*h,)

                    row_idx.append(rows_block)
                    col_idx.append(cols_block)
                    data_list.append(data_block)

            logger.debug("Built H contributions for order %s.", order)

        if data_list:
            all_rows = np.concatenate(row_idx)
            all_cols = np.concatenate(col_idx)
            all_data = np.concatenate(data_list)
        else:
            all_rows = np.array([], dtype=np.intp)
            all_cols = np.array([], dtype=np.intp)
            all_data = np.array([], dtype=float)

        H = csr_matrix(
            (all_data, (all_rows, all_cols)),
            shape=(n_pix, n_pix * h),
        )
        logger.debug(
            "SciPySparseOperator built: H shape %s, nnz=%d", H.shape, H.nnz
        )
        return cls(H, image_shape)

    @classmethod
    def load(cls, path: Path) -> "SciPySparseOperator":
        """Load a saved operator from an ``.npz`` file.

        Parameters
        ----------
        path : Path
            File written by :meth:`save`.

        Returns
        -------
        SciPySparseOperator
        """
        archive = np.load(path, allow_pickle=False)
        H = csr_matrix(
            (archive["data"], archive["indices"], archive["indptr"]),
            shape=tuple(archive["h_shape"]),
        )
        image_shape = tuple(int(x) for x in archive["image_shape"])
        return cls(H, image_shape)  # type: ignore[arg-type]

    def save(self, path: Path) -> None:
        """Save the operator to a single ``.npz`` file.

        Parameters
        ----------
        path : Path
            Output path. The ``.npz`` extension is added if absent.
        """
        H = self._H.tocsr()
        np.savez(
            path,
            data=H.data,
            indices=H.indices,
            indptr=H.indptr,
            h_shape=np.array(H.shape),
            image_shape=np.array(self.image_shape),
        )
        logger.debug("Saved SciPySparseOperator to %s.", path)

    def apply(self, a_tilde: np.ndarray) -> np.ndarray:
        """Forward pass: ``H @ a_tilde``.

        Parameters
        ----------
        a_tilde : np.ndarray
            Shape ``(n_coefficients,)``.

        Returns
        -------
        np.ndarray
            Shape ``(n_rows * n_cols,)``.
        """
        return np.asarray(self._H @ a_tilde)

    def apply_adjoint(self, f: np.ndarray) -> np.ndarray:
        """Adjoint pass: ``H.T @ f``.

        Parameters
        ----------
        f : np.ndarray
            Shape ``(n_rows * n_cols,)``.

        Returns
        -------
        np.ndarray
            Shape ``(n_coefficients,)``.
        """
        return np.asarray(self._H.T @ f)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest unittests/test_operator.py -v
```

Expected: all 11 tests PASS.

- [ ] **Step 5: Run ruff and ty**

```bash
uv run ruff check src/spectrex/operator.py
uv run ty check src/spectrex/operator.py
```

Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/spectrex/operator.py unittests/test_operator.py
git commit -m "feat: add ForwardOperatorProtocol and SciPySparseOperator"
```

---

## Task 4: `NoiseModel` and `SpectralSolver` (solver.py)

**Files:**
- Create: `src/spectrex/solver.py`
- Create: `unittests/test_solver.py`

- [ ] **Step 1: Write the failing tests in `unittests/test_solver.py`**

```python
"""Unit tests for spectrex.solver."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import eye as speye

from spectrex.operator import SciPySparseOperator
from spectrex.solver import NoiseModel, SpectralSolver


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def identity_operator() -> SciPySparseOperator:
    """Identity operator: H = I_6, image_shape=(2,3), n_coefficients=6."""
    H = speye(6, format="csr")
    return SciPySparseOperator(H, image_shape=(2, 3))


@pytest.fixture
def rectangular_operator() -> SciPySparseOperator:
    """H is (4, 6): first 4 rows of the 6x6 identity."""
    H = speye(6, format="csr")[:4, :]
    return SciPySparseOperator(H, image_shape=(2, 2))


# ---------------------------------------------------------------------------
# NoiseModel
# ---------------------------------------------------------------------------

def test_variance_nonneg():
    nm = NoiseModel(read_noise=5.0)
    f = np.array([-1000.0, -1.0, 0.0, 10.0, 100.0])
    assert np.all(nm.variance(f) >= 0.0)


def test_variance_floor():
    nm = NoiseModel(read_noise=5.0)
    f = np.array([0.0])
    np.testing.assert_allclose(nm.variance(f), [25.0])  # 0 + 5^2


def test_variance_poisson_plus_readnoise():
    nm = NoiseModel(read_noise=3.0)
    f = np.array([16.0])
    np.testing.assert_allclose(nm.variance(f), [25.0])  # 16 + 9


def test_precision_weights_positive():
    nm = NoiseModel(read_noise=5.0)
    f = np.array([0.0, 25.0, -50.0])
    assert np.all(nm.precision_weights(f) > 0.0)


def test_precision_weights_formula():
    nm = NoiseModel(read_noise=5.0)
    f = np.array([0.0, 75.0])
    # variance = [25, 100] -> weights = [1/5, 1/10]
    expected = np.array([1.0 / 5.0, 1.0 / 10.0])
    np.testing.assert_allclose(nm.precision_weights(f), expected)


# ---------------------------------------------------------------------------
# SpectralSolver.solve — identity operator
# ---------------------------------------------------------------------------

def test_solve_identity_exact(identity_operator):
    """H = I: solution should equal the RHS up to solver tolerance."""
    f = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    solver = SpectralSolver(identity_operator)
    result = solver.solve(f.reshape(2, 3))
    np.testing.assert_allclose(result, f, atol=1e-6)


def test_solve_output_shape(identity_operator):
    f = np.ones(6)
    solver = SpectralSolver(identity_operator)
    result = solver.solve(f.reshape(2, 3))
    assert result.shape == (6,)


# ---------------------------------------------------------------------------
# SpectralSolver.solve — support mask
# ---------------------------------------------------------------------------

def test_solve_mask_zeros_excluded(identity_operator):
    """Columns excluded by the mask must be zero in the output."""
    f = np.ones(6)
    mask = np.array([True, False, True, False, True, False])
    solver = SpectralSolver(identity_operator)
    result = solver.solve(f.reshape(2, 3), support_mask=mask)
    np.testing.assert_array_equal(result[[1, 3, 5]], 0.0)


def test_solve_mask_active_nonzero(identity_operator):
    """Active columns in the mask should recover the signal."""
    f = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    mask = np.array([True, False, True, False, True, False])
    solver = SpectralSolver(identity_operator)
    result = solver.solve(f.reshape(2, 3), support_mask=mask)
    np.testing.assert_allclose(result[[0, 2, 4]], 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# SpectralSolver.solve_regularised
# ---------------------------------------------------------------------------

def test_solve_regularised_output_shape(identity_operator):
    f = np.ones(6)
    solver = SpectralSolver(identity_operator, regularisation=1e-2)
    result = solver.solve_regularised(f.reshape(2, 3))
    assert result.shape == (6,)


def test_solve_regularised_finite(identity_operator):
    rng = np.random.default_rng(42)
    f = rng.standard_normal(6)
    solver = SpectralSolver(
        identity_operator,
        noise_model=NoiseModel(read_noise=5.0),
        regularisation=1e-2,
    )
    result = solver.solve_regularised(f.reshape(2, 3))
    assert np.all(np.isfinite(result))


def test_solve_regularised_reduces_residual(rectangular_operator):
    """Regularised result should fit the data better than all-zeros."""
    rng = np.random.default_rng(7)
    f = rng.standard_normal(4)
    solver = SpectralSolver(rectangular_operator, regularisation=1e-3)
    result = solver.solve_regularised(f.reshape(2, 2))
    residual = np.linalg.norm(rectangular_operator.apply(result) - f)
    assert residual < np.linalg.norm(f)


def test_solve_regularised_with_noise_model(rectangular_operator):
    rng = np.random.default_rng(11)
    f = np.abs(rng.standard_normal(4)) * 100  # positive counts
    solver = SpectralSolver(
        rectangular_operator,
        noise_model=NoiseModel(read_noise=5.0),
        regularisation=1e-2,
    )
    result = solver.solve_regularised(f.reshape(2, 2))
    assert np.all(np.isfinite(result))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest unittests/test_solver.py -v 2>&1 | head -20
```

Expected: `ImportError: cannot import name 'NoiseModel' from 'spectrex.solver'`.

- [ ] **Step 3: Create `src/spectrex/solver.py`**

```python
"""Spectral recovery solvers for grism WFSS deconvolution."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import LinearOperator, lsmr, lsqr

from spectrex.operator import ForwardOperatorProtocol

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NoiseModel:
    """Poisson + read-noise model for JWST NIRISS detectors.

    Parameters
    ----------
    read_noise : float
        Detector read noise in electrons. Default 5.0.
    """

    read_noise: float = 5.0

    def variance(self, f: np.ndarray) -> np.ndarray:
        """Per-pixel variance: ``σ²(f) = max(f, 0) + read_noise²``.

        Parameters
        ----------
        f : np.ndarray
            Observed pixel values (may be negative after sky subtraction).

        Returns
        -------
        np.ndarray
            Non-negative variance, same shape as ``f``.
        """
        return np.maximum(f, 0.0) + self.read_noise**2

    def precision_weights(self, f: np.ndarray) -> np.ndarray:
        """Precision weights ``1 / σ(f)`` for whitening the linear system.

        Parameters
        ----------
        f : np.ndarray
            Observed pixel values.

        Returns
        -------
        np.ndarray
            Positive weight array, same shape as ``f``.
        """
        return 1.0 / np.sqrt(self.variance(f))


class SpectralSolver:
    """Least-squares solver for WFSS spectral deconvolution.

    Parameters
    ----------
    operator : ForwardOperatorProtocol
        The grism forward operator H. Any implementation satisfying
        the protocol is accepted (scipy or future JAX).
    noise_model : NoiseModel, optional
        Noise model for whitening in :meth:`solve_regularised`.
        Uses uniform weights if ``None``.
    regularisation : float
        Tikhonov regularisation parameter λ for
        :meth:`solve_regularised`. Default 1e-2.
    max_iter : int
        Maximum solver iterations. Default 1000.
    tolerance : float
        Convergence tolerance (``atol`` and ``btol``). Default 1e-10.
    """

    def __init__(
        self,
        operator: ForwardOperatorProtocol,
        noise_model: NoiseModel | None = None,
        regularisation: float = 1e-2,
        max_iter: int = 1000,
        tolerance: float = 1e-10,
    ) -> None:
        self._operator = operator
        self._noise_model = noise_model
        self._regularisation = regularisation
        self._max_iter = max_iter
        self._tolerance = tolerance

    def solve(
        self,
        dispersed: np.ndarray,
        support_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """LSQR solve for source coefficients.

        Minimises ``||H a - f||²``.

        Parameters
        ----------
        dispersed : np.ndarray
            Dispersed detector image, shape ``image_shape``.
        support_mask : np.ndarray, optional
            Boolean array, shape ``(n_coefficients,)``. When provided,
            the solve is restricted to ``True`` columns; the returned
            vector has zeros elsewhere.

        Returns
        -------
        np.ndarray
            Coefficient vector ``a_tilde``, shape ``(n_coefficients,)``.
        """
        f = dispersed.ravel().astype(float)
        n_pix = f.size
        n_coef = self._operator.n_coefficients

        if support_mask is not None:
            active_idx = np.where(support_mask)[0]
            n_active = len(active_idx)

            def _matvec(v: np.ndarray) -> np.ndarray:
                full = np.zeros(n_coef)
                full[active_idx] = v
                return self._operator.apply(full)

            def _rmatvec(v: np.ndarray) -> np.ndarray:
                return self._operator.apply_adjoint(v)[active_idx]

            A = LinearOperator(
                shape=(n_pix, n_active),
                matvec=_matvec,
                rmatvec=_rmatvec,
                dtype=float,
            )
            res = lsqr(
                A, f,
                iter_lim=self._max_iter,
                atol=self._tolerance,
                btol=self._tolerance,
            )
            d = np.zeros(n_coef)
            d[active_idx] = res[0]
        else:

            def _matvec2(v: np.ndarray) -> np.ndarray:
                return self._operator.apply(v)

            def _rmatvec2(v: np.ndarray) -> np.ndarray:
                return self._operator.apply_adjoint(v)

            A = LinearOperator(
                shape=(n_pix, n_coef),
                matvec=_matvec2,
                rmatvec=_rmatvec2,
                dtype=float,
            )
            res = lsqr(
                A, f,
                iter_lim=self._max_iter,
                atol=self._tolerance,
                btol=self._tolerance,
            )
            d = res[0]

        logger.debug("solve: itn=%d r1norm=%.3e", res[2], res[3])
        return d

    def solve_regularised(
        self,
        dispersed: np.ndarray,
    ) -> np.ndarray:
        """LSMR solve with Tikhonov regularisation and noise weighting.

        Minimises ``||W (H a - f)||² + λ ||a||²``
        where ``W = diag(1/σ)`` from ``self.noise_model``
        (identity if ``None``).

        Parameters
        ----------
        dispersed : np.ndarray
            Dispersed detector image, shape ``image_shape``.

        Returns
        -------
        np.ndarray
            Coefficient vector ``a_tilde``, shape ``(n_coefficients,)``.
        """
        f = dispersed.ravel().astype(float)
        n_pix = f.size
        n_coef = self._operator.n_coefficients

        if self._noise_model is not None:
            w = self._noise_model.precision_weights(f)
        else:
            w = np.ones(n_pix)

        def _matvec_w(v: np.ndarray) -> np.ndarray:
            return w * self._operator.apply(v)

        def _rmatvec_w(v: np.ndarray) -> np.ndarray:
            return self._operator.apply_adjoint(w * v)

        sqrt_lam = float(np.sqrt(self._regularisation))

        def _matvec_reg(v: np.ndarray) -> np.ndarray:
            return np.concatenate([_matvec_w(v), sqrt_lam * v])

        def _rmatvec_reg(v: np.ndarray) -> np.ndarray:
            return _rmatvec_w(v[:n_pix]) + sqrt_lam * v[n_pix:]

        A_reg = LinearOperator(
            shape=(n_pix + n_coef, n_coef),
            matvec=_matvec_reg,
            rmatvec=_rmatvec_reg,
            dtype=float,
        )
        f_reg = np.concatenate([w * f, np.zeros(n_coef)])

        res = lsmr(
            A_reg,
            f_reg,
            atol=self._tolerance,
            btol=self._tolerance,
            maxiter=self._max_iter,
        )
        logger.debug("solve_regularised: itn=%d normr=%.3e", res[2], res[3])
        return res[0]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest unittests/test_solver.py -v
```

Expected: all 15 tests PASS.

- [ ] **Step 5: Run ruff and ty**

```bash
uv run ruff check src/spectrex/solver.py
uv run ty check src/spectrex/solver.py
```

Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/spectrex/solver.py unittests/test_solver.py
git commit -m "feat: add NoiseModel and SpectralSolver with LSQR/LSMR and Tikhonov"
```

---

## Task 5: Wire up the public API (`__init__.py`)

**Files:**
- Modify: `src/spectrex/__init__.py`

- [ ] **Step 1: Update `src/spectrex/__init__.py`**

```python
"""specTrex — Grism spectra extraction in crowded regions.

Public API
----------
InstrumentConfig
    Instrument configuration loader (grism, filter, sensitivity curves).
EigenspectraBasis
    PCA eigenspectra basis for spectral representation.
ForwardOperatorProtocol
    Protocol that any forward operator must satisfy.
SciPySparseOperator
    scipy-sparse forward operator (Phase 1).
NoiseModel
    Poisson + read-noise model.
SpectralSolver
    LSQR/LSMR solver for grism deconvolution.
"""

from spectrex._version import version as __version__
from spectrex.basis import EigenspectraBasis
from spectrex.instrument import InstrumentConfig
from spectrex.operator import ForwardOperatorProtocol, SciPySparseOperator
from spectrex.solver import NoiseModel, SpectralSolver

__all__ = [
    "__version__",
    "InstrumentConfig",
    "EigenspectraBasis",
    "ForwardOperatorProtocol",
    "SciPySparseOperator",
    "NoiseModel",
    "SpectralSolver",
]
```

- [ ] **Step 2: Verify imports work from the package root**

```bash
uv run python -c "
import spectrex
print(spectrex.__version__)
print(spectrex.InstrumentConfig)
print(spectrex.EigenspectraBasis)
print(spectrex.ForwardOperatorProtocol)
print(spectrex.SciPySparseOperator)
print(spectrex.NoiseModel)
print(spectrex.SpectralSolver)
"
```

Expected: version string printed, all six class names printed, no errors.

- [ ] **Step 3: Run the full fast test suite**

```bash
uv run pytest unittests/ -v -m "not slow"
```

Expected: all tests PASS, no errors.

- [ ] **Step 4: Run ruff and ty on the whole package**

```bash
uv run ruff check src/spectrex/
uv run ty check src/spectrex/
```

Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add src/spectrex/__init__.py
git commit -m "feat: wire up public API in spectrex/__init__.py"
```

---

## Task 6: Slow integration tests

**Files:**
- Create: `unittests/test_integration.py`

These tests use real `testdata/` files and `image_shape=(500, 20)`. They are marked `@pytest.mark.slow` and skipped in the default CI run.

- [ ] **Step 1: Write `unittests/test_integration.py`**

```python
"""Integration tests using real testdata files.

Run with:  pytest unittests/test_integration.py -v -m slow
Skip with: pytest unittests/ -m "not slow"
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spectrex import (
    EigenspectraBasis,
    InstrumentConfig,
    SciPySparseOperator,
    SpectralSolver,
)


@pytest.fixture(scope="module")
def testdata_dir() -> Path:
    return Path(__file__).parent.parent / "testdata"


@pytest.fixture(scope="module")
def config(testdata_dir: Path) -> InstrumentConfig:
    return InstrumentConfig.from_files(
        conf_path=testdata_dir / "Config Files" / "GR150R.F150W.220725.conf",
        wavelengthrange_path=testdata_dir / "jwst_niriss_wavelengthrange_0002.asdf",
        sensitivity_dir=testdata_dir / "SenseConfig" / "wfss-grism-configuration",
        filter_name="F150W",
        n_wavelengths=150,
    )


@pytest.fixture(scope="module")
def basis(testdata_dir: Path, config: InstrumentConfig) -> EigenspectraBasis:
    return EigenspectraBasis.from_csv(
        testdata_dir / "eigenspectra_kurucz.csv",
        config.wavelengths,
    )


@pytest.mark.slow
def test_config_loads(config):
    assert config.grism == "GR150R"
    assert config.filter_name == "F150W"
    assert config.wavelengths.shape == (150,)
    assert set(config.orders) >= {"A", "B", "C"}


@pytest.mark.slow
def test_basis_loads(basis):
    assert basis.n_components == 10
    assert basis.components.shape == (150, 10)


@pytest.mark.slow
def test_operator_build_shape(config, basis):
    image_shape = (500, 20)
    op = SciPySparseOperator.build(config, basis, image_shape)
    n_pix = 500 * 20
    assert op.image_shape == image_shape
    assert op.n_coefficients == n_pix * 10
    assert op._H.shape == (n_pix, n_pix * 10)


@pytest.mark.slow
def test_operator_save_load_roundtrip(config, basis, tmp_path):
    op = SciPySparseOperator.build(config, basis, image_shape=(50, 20))
    path = tmp_path / "op.npz"
    op.save(path)
    loaded = SciPySparseOperator.load(path)
    rng = np.random.default_rng(0)
    a = rng.standard_normal(op.n_coefficients)
    np.testing.assert_allclose(loaded.apply(a), op.apply(a), rtol=1e-12)


@pytest.mark.slow
def test_forward_apply_shape(config, basis):
    image_shape = (50, 20)
    op = SciPySparseOperator.build(config, basis, image_shape)
    rng = np.random.default_rng(1)
    a_tilde = rng.standard_normal(op.n_coefficients)
    f = op.apply(a_tilde)
    assert f.shape == (50 * 20,)


@pytest.mark.slow
def test_roundtrip_recovery(config, basis):
    """Disperse a sparse mock scene then recover; check residual is small."""
    image_shape = (50, 20)
    op = SciPySparseOperator.build(config, basis, image_shape)

    # Build a sparse a_tilde: only 3 active pixels
    rng = np.random.default_rng(42)
    n_pix = 50 * 20
    a_tilde = np.zeros(n_pix * basis.n_components)
    for k in [100, 250, 600]:
        # Find a coefficient set that gives positive flux
        for _ in range(100):
            candidate = rng.standard_normal(basis.n_components)
            if np.all(basis.reconstruct(candidate) >= 0):
                a_tilde[k * basis.n_components:(k + 1) * basis.n_components] = candidate
                break

    dispersed = op.apply(a_tilde).reshape(image_shape)

    # Build support mask from true a_tilde support
    mask = np.abs(a_tilde) > 0
    solver = SpectralSolver(op, max_iter=500, tolerance=1e-8)
    recovered = solver.solve(dispersed, support_mask=mask)

    # Recovered coefficients should closely match the input
    active = mask
    np.testing.assert_allclose(
        recovered[active], a_tilde[active], rtol=1e-3, atol=1e-6
    )
```

- [ ] **Step 2: Verify slow tests are skipped by default**

```bash
uv run pytest unittests/ -v -m "not slow" --co 2>&1 | grep "test_integration"
```

Expected: no `test_integration` tests collected.

- [ ] **Step 3: Run all fast tests one final time to confirm nothing is broken**

```bash
uv run pytest unittests/ -v -m "not slow"
```

Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
git add unittests/test_integration.py
git commit -m "test: add slow integration tests with real testdata and round-trip recovery"
```

---

## Final verification

- [ ] **Run the full fast suite and linters**

```bash
uv run pytest unittests/ -m "not slow" -v
uv run ruff check src/spectrex/
uv run ty check src/spectrex/
```

Expected: all tests PASS, no lint or type errors.

- [ ] **Verify package installs cleanly**

```bash
uv pip install -e . --quiet
uv run python -c "from spectrex import SpectralSolver; print('OK')"
```

Expected: `OK`.
