# Design: Mock Example Notebook

**Date:** 2026-05-05  
**Status:** Approved  
**Author:** Jarvis

---

## Overview

A Jupyter notebook demonstrating the end-to-end `spectrex` Phase 1 API using a synthetic (mock) scene. It replaces `legacy/Ex/playground_mock.py` with clean, idiomatic usage of the new public API. The notebook serves two audiences: developers running it locally and users reading the rendered Sphinx docs.

---

## File Layout

```
notebooks/
  mock_example.ipynb          ← canonical notebook
  operator_cache.npz          ← gitignored; built on first run

docs/content/
  mock_example.ipynb          ← symlink → ../../notebooks/mock_example.ipynb
```

- `operator_cache.npz` is added to `.gitignore`.
- The symlink allows `myst-nb` to render the notebook at the docs URL with no duplication.
- `docs/index.rst` (or equivalent toctree) gains an entry pointing to `content/mock_example`.

---

## Notebook Structure

Single linear narrative notebook. No helper functions — all API calls are inline so the reader sees the full pipeline at each step. Structure below maps to numbered cells.

### Constants block (cell 2)

```python
TESTDATA       = Path("../testdata")
OPERATOR_CACHE = Path("operator_cache.npz")
COLD_START     = False   # set True to force operator rebuild from scratch
IMAGE_SHAPE    = (500, 20)
SOURCE_DENSITY = 0.1     # fraction of pixels with injected sources
SEED           = 42
N_COMPONENTS   = 10      # must match eigenspectra CSV
```

### Cell sequence

| Cell | Type     | Content |
|------|----------|---------|
| 1    | Markdown | Title + one-paragraph description of what the notebook demonstrates |
| 2    | Code     | Imports + constants (above) |
| 3    | Markdown | **Step 1 — Instrument configuration & eigenspectra basis** |
| 4    | Code     | `InstrumentConfig.from_files(...)`, `EigenspectraBasis.from_csv(...)` |
| 5    | Markdown | **Step 2 — Forward operator** — note that first run may take ~N seconds; set `COLD_START=True` to rebuild |
| 6    | Code     | `if OPERATOR_CACHE.exists() and not COLD_START: load` else `build + save` |
| 7    | Markdown | **Step 3 — Mock scene** — description of `random_stars_PCA` construction |
| 8    | Code     | Generate `a_tilde` (sparse PCA coefficients, 10% source density); compute direct image via `basis.broadband_image` |
| 9    | Markdown | **§1 — Noiseless recovery** |
| 10   | Code     | Forward model: `op.apply(a_tilde)` → `dispersed`; build `support_mask` from direct image |
| 11   | Code     | `SpectralSolver(op).solve(dispersed, support_mask=support_mask)` → `recovered` |
| 12   | Code     | Collect `true_flux` / `rec_flux` arrays via `basis.reconstruct`; compute flux RMSE over active pixels |
| 13   | Code     | 4-panel figure: Direct \| Dispersed \| Recovered \| \|Residual\| |
| 14   | Code     | Parity plot: `true_flux` vs `rec_flux`, RMSE annotated |
| 15   | Markdown | **§2 — Noisy recovery** |
| 16   | Code     | `NoiseModel(read_noise=5).sample(dispersed, rng)` → `noisy_dispersed` |
| 17   | Code     | `SpectralSolver(op).solve(noisy_dispersed, support_mask=support_mask, noise_model=noise_model)` → `noisy_recovered` |
| 18   | Code     | Collect noisy flux arrays; compute noisy RMSE |
| 19   | Code     | 4-panel figure (noisy) |
| 20   | Code     | Parity plot (noisy) with RMSE annotation |
| 21   | Markdown | **Next steps** — note pointing to future `analysis_rmse_vs_density.ipynb` for the full RMSE-vs-source-density sweep |

---

## Figures

### 4-panel image figure (cells 13, 19)

- `figsize=(14, 4)`, tight layout
- All panels: `cmap='inferno'`, `origin='lower'`, `aspect='auto'`, individual colorbars
- **Direct** and **Recovered** panels share the same `vmin`/`vmax` computed as `mean ± 2σ` of the direct image
- **Dispersed** panel: `vmin`/`vmax` from its own `mean ± 2σ`
- **Residual** panel (`|Direct − Recovered|`): `vmin=0`, `vmax=mean + 1σ` of the residual
- Panel titles: "Direct image", "Dispersed (grism)", "Recovered", "|Residual|"

### Parity plot (cells 14, 20)

- `figsize=(5, 5)`, equal aspect ratio
- Scatter of `true_flux` vs `rec_flux` across all wavelength samples for all active pixels: `s=2, alpha=0.05`
- Red dashed 1:1 reference line
- RMSE annotated as `f"RMSE = {rmse:.4f}"` in upper-left corner via `ax.text`
- Axis labels: `"True flux f(λ)"` and `"Recovered flux f(λ)"`

### RMSE definition

Computed on reconstructed flux values, not raw PCA coefficients:

```python
active_indices = [k for k in range(n_pix) if np.any(a_tilde[k*n:(k+1)*n] != 0)]
true_flux  = np.concatenate([basis.reconstruct(a_tilde[k*n:(k+1)*n]) for k in active_indices])
rec_flux   = np.concatenate([basis.reconstruct(recovered[k*n:(k+1)*n]) for k in active_indices])
rmse = np.sqrt(np.mean((true_flux - rec_flux) ** 2))
```

This matches the parity plot axes and is physically meaningful (flux units).

---

## Operator Caching Pattern

```python
if OPERATOR_CACHE.exists() and not COLD_START:
    op = SciPySparseOperator.load(OPERATOR_CACHE)
else:
    import time
    t0 = time.perf_counter()
    op = SciPySparseOperator.build(config, basis, IMAGE_SHAPE)
    op.save(OPERATOR_CACHE)
    print(f"Operator built in {time.perf_counter() - t0:.1f} s — cached to {OPERATOR_CACHE}")
```

A markdown cell immediately before this code block notes that the first run will take time and explains `COLD_START`.

---

## Future Work (out of scope here)

A separate file `notebooks/analysis_rmse_vs_density.ipynb` (or `.py` script) will:

- Define a `run_pipeline(a_tilde, op, basis, noise_model=None)` helper
- Sweep `SOURCE_DENSITY` over ~5–8 values on the full 500×20 image
- Plot RMSE vs source density for noiseless and noisy cases
- This sweep is important for establishing method credibility

---

## Constraints

- `myst-nb` is configured with `nb_execution_mode = "off"` — the notebook is **not** executed at doc build time. The notebook must be pre-executed and committed with outputs so the rendered docs are readable. The `operator_cache.npz` file does not need to be committed; only the notebook outputs (figures, printed text) need to be present in the `.ipynb` JSON.
- `notebooks/operator_cache.npz` must be in `.gitignore`.
- `NoiseModel` requires a new `sample(image, rng)` method (3 lines in `solver.py`) that draws `image + rng.normal(0, sqrt(variance(image)))`. This is a minor API addition included in the implementation plan.
- No new dependencies beyond what is already in `pyproject.toml` (`matplotlib` assumed available in the dev environment; add to `[dependency-groups].dev` if not already present).
