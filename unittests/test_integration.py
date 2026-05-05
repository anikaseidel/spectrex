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
                a_tilde[
                    k * basis.n_components:(k + 1) * basis.n_components
                ] = candidate
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
