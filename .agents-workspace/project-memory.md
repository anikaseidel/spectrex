# specTrex — Project Memory

## Stack
- Python 3.12+ via `uv`; `uv run python` / `uv run pytest`
- Linting: `ruff`; type checking: `ty`
- Library only (no CLI)
- `numpy>=2,<3`; `jax>=0.10` (hard dep Phase 2+); `scipy>=1.17`
- Tests: `pytest`; slow tests marked `@pytest.mark.slow`, skipped in default CI
- Notebooks: pre-executed, outputs committed; `myst-nb nb_execution_mode = "off"`

## Layout
- `src/spectrex/` — package source
- `unittests/` — pytest suite
- `notebooks/` — Jupyter notebooks (pre-executed)
- `docs/` — Sphinx docs; `docs/content/` symlinks to `notebooks/`
- `testdata/` — calibration data, not shipped in wheel
- `.worktrees/` — git worktrees for feature branches

## Key Conventions
- All file paths explicit (no CWD assumptions)
- `numpy>=2`: use `np.trapezoid`, not `np.trapz`
- Replace `pandas` / `scipy.interpolate.interp1d` with `np.genfromtxt` / `np.interp`
- `@dataclass(frozen=True)` for value objects; `EigenspectraBasis` arrays read-only
- `_make_linear_op()` subclass pattern for `ty` compatibility
- `ForwardOperatorProtocol` is `@runtime_checkable`
- Package logger: `_logging` alias + `NullHandler` in `__init__.py`

## Architecture Decisions
- `InstrumentConfig`: regular class (not `@dataclass`) — `GrismTrace` not hashable
- `SciPySparseOperator.save/load`: single `.npz` storing CSR triplets
- `NoiseModel.sample()`: `(f + rng.normal(0.0, sigma)).astype(f.dtype)` — `.astype` outside sum
- **Phase 2 JAX operator**: compact trace structure:
  - `trace_indices[K, O, L]` int32 — pixel index per source/order/wavelength
  - `weights[O, L, M]` float32 — shared sensitivity × basis (54 KB, image-size-independent)
  - Ghost pixel at `n_pix` absorbs out-of-bounds wavelengths in `apply`; adjoint pads f with 0
  - `apply`: `einsum('km,olm->kol', a, W)` then `scatter_add` to ghost-pixel-extended image
  - `apply_adjoint`: gather into `f_padded[trace_indices]` then `einsum('kol,olm->km', ...)`
- **Phase 2 solver**: FISTA (O(1/k²)) with group-L1 `λ Σ_k ||a_k||₂` (block soft-threshold)
  - Step size from power iteration on `H^T W^2 H`; noise weights from `NoiseModel.precision_weights()`
- **Support mask (Phase 2)**: baked into `JAXOperator` at build time (source catalog always available from direct image in WFSS workflow)
- **Phase 3 (future)**: iterative catalog refinement — solve → detect new sources in residuals → rebuild → repeat; no fundamental JAXOperator redesign needed
- **Full-blind extraction**: Phase 3b / research paper territory; not planned for this package

## Phase Status
- **Phase 1**: COMPLETE — merged to main, tagged v0.1.0
- **Mock notebook**: COMPLETE — merged
- **Phase 2**: COMPLETE — merged to main, tagged v0.2.0
- **Analysis notebook**: COMPLETE — merged
- **Comparison notebooks (v2.1)**: COMPLETE — merged; two new notebooks on main
- **API documentation**: COMPLETE — 5 RST pages, 4 graphviz diagrams, 2 embedded PNGs; build passes with zero new warnings
- **instrument_primer notebook**: COMPLETE — §0–§5, pre-executed, integrated into Sphinx docs; tagged v0.2.2
- **Phase 3**: PARKED — three candidate directions documented in `docs/superpowers/specs/2026-05-06-phase3-directions.md`

## Phase 3 Directions (undecided)
- **A — Uncertainty quantification**: credible intervals on reconstructed spectra; support-conditioned Gaussian as baseline, MCMC/VI as stretch
- **B — Full-detector scaling**: tiling strategy for 2048×2048 NIRISS frame; ~10k sources; overlapping trace contamination
- **C — Source detection / weak positional prior**: iterative catalog refinement (C1, lowest risk) through joint position+spectrum optimisation (C2) and Order-B-anchored detection (C4); C1 buildable on Phase 2 with modest effort; B is practical prerequisite for full-field C

## Docs Notes
- `sphinx.ext.graphviz` added to conf.py; graphviz 14.1.2 installed via `pixi global install graphviz` (binary at `~/.pixi/bin/dot`); `dot -c` must be run once to register plugins
- `ForwardOperatorProtocol` documented with manual `.. py:class::` NOT `.. autoclass::` — sphinx autodoc + typing.Protocol creates duplicate object descriptions for abstract members
- Figures extracted from pre-executed notebook outputs via `scripts/extract_nb_figures.py` pattern (inline Python, not committed)

## Test Counts (main, fast suite)
- 70 tests passing (fast); 8 slow tests

## API Corrections (confirmed from execution)
- `InstrumentConfig.from_files(conf_path, wavelengthrange_path, sensitivity_dir, filter_name)` — NOT `from_config_dir`
- `JAXProximalSolver(op, noise_model, lam, max_iter)` — parameter is `max_iter`, NOT `n_iter`
- `SciPySparseOperator` internal matrix attribute: `._H` — NOT `._matrix`
- `EigenspectraBasis.basis.components` shape: `(n_wav, M)` — reconstruction: `basis.components @ c` (NOT `.T @`)
- `IMAGE_SHAPE` for notebooks: use `(500, 20)` not `(50, 20)` — traces go out of bounds on 50-row images

## Grism Geometry — GR150R / F150W, (500 × 20) stamp

Order letter → diffraction order integer: `A=1` (first), `B=0` (zeroth), `C=2` (second)

For source at `(row i, col j)`:
- **Order A** (first, dispersed): x_trace = i + 57 to i + 147, y_trace ≈ j − 1.15 → 91 unique pixels, ~1.6 λ/pixel
- **Order B** (zeroth, undispersed): x_trace = i − 216 to i − 213, y_trace ≈ j − 2.3 → 4–5 unique pixels, ~35 λ/pixel
- **Order C** (second): x_trace = i + 324 to i + 501 → mostly out-of-bounds for 500-row stamp

**Individual bright pixels in the dispersed image = zeroth-order (Order B) spots.**
- Same total sensitivity as Order A (~0.12) but concentrated into 4–5 pixels instead of 91
- Yields ~19–23× higher flux per pixel compared to first-order
- Located ~216 rows below and ~2 columns left of the source
- Confirmed: pixel (38,13) receives 30–43 λ each from sources at rows 251–254, col 15 via Order B
- This is expected physical behavior (zeroth-order contamination) — not a bug

## Critical Notes
- Operator build for 500×20 ~60 s; notebooks use `COLD_START` flag + cache `.npz`
- `ty` does not recognise `LinearOperator.__init__` keyword args — use subclass
- `matplotlib>=3.10.8` already in core deps
- `jax` already in core deps (added in Phase 1 pyproject.toml)
- `optax` NOT needed for Phase 2 FISTA (step size is analytical; optax deferred to Phase 3)
- JAX 0.10.0 confirmed available in venv
- Main is 11 commits ahead of `origin/main`
