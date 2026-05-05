# Mock Example Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce `notebooks/mock_example.ipynb` — a pre-executed Jupyter notebook demonstrating the full spectrex Phase 1 API (build → disperse → recover) on a synthetic mock scene, with noiseless and noisy recovery sections, lightly polished figures, and RMSE-annotated parity plots. Also add `NoiseModel.sample()`, set up the package-level logger, hook the notebook into the Sphinx docs.

**Architecture:** Linear narrative notebook (no helper functions), operator cached via `COLD_START` flag, two independent recovery sections (noiseless then noisy) each producing a 4-panel image figure and a parity plot. A symlink from `docs/content/` lets myst-nb render the notebook without duplication. `NoiseModel` gains a `sample()` method (3 lines). Package `__init__.py` gains the root logger + `NullHandler`.

**Tech Stack:** Python 3.12+, `spectrex` public API, `numpy`, `matplotlib`, `jupyter`/`nbformat` (dev dependency), `myst-nb` (already configured), `scipy`.

---

## File Map

| Action | Path | Purpose |
|--------|------|---------|
| Create | `notebooks/mock_example.ipynb` | Canonical notebook |
| Create | `docs/content/mock_example.ipynb` | Symlink → `../../notebooks/mock_example.ipynb` |
| Modify | `docs/index.rst` | Add Examples toctree entry |
| Modify | `src/spectrex/solver.py` | Add `NoiseModel.sample()` |
| Modify | `src/spectrex/__init__.py` | Add root logger + NullHandler |
| Modify | `pyproject.toml` | Add `jupyter`, `nbformat` to `[dependency-groups].dev` |
| Modify | `.gitignore` | Add `notebooks/operator_cache.npz` |
| Modify | `unittests/test_solver.py` | Add test for `NoiseModel.sample()` |

---

### Task 0: Pre-flight — dev dependencies and .gitignore

**Files:**
- Modify: `pyproject.toml`
- Modify: `.gitignore`

- [ ] **Step 1: Add jupyter and nbformat to dev dependencies**

Edit `pyproject.toml` `[dependency-groups].dev`:

```toml
dev = [
    "pytest>=9.0.3",
    "ty>=0.0.34",
    "jupyter>=1.1",
    "nbformat>=5.10",
]
```

- [ ] **Step 2: Add operator cache to .gitignore**

Add to `.gitignore`:

```
notebooks/operator_cache.npz
```

- [ ] **Step 3: Sync the venv**

```bash
uv sync
```

Expected: packages install without errors.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml .gitignore
git commit -m "chore: add jupyter/nbformat dev deps; gitignore operator cache"
```

---

### Task 1: Package-level logger

**Files:**
- Modify: `src/spectrex/__init__.py`

- [ ] **Step 1: Verify current __init__.py content**

```bash
cat src/spectrex/__init__.py
```

The file currently imports from submodules and exposes `__all__`. It does not set up a root logger.

- [ ] **Step 2: Add root logger**

After the existing imports block, add:

```python
import logging as _logging

# Standard library best practice for packages: attach a NullHandler so the
# library never emits "No handlers could be found" warnings when the caller
# hasn't configured logging. Users configure the root "spectrex" logger to
# capture log output from all submodules.
_logging.getLogger("spectrex").addHandler(_logging.NullHandler())

#: Package-level logger.  Users can configure it directly::
#:
#:     import logging, spectrex
#:     logging.getLogger("spectrex").setLevel(logging.DEBUG)
logger: _logging.Logger = _logging.getLogger("spectrex")
```

Also add `"logger"` to `__all__`.

- [ ] **Step 3: Verify import works**

```bash
uv run python -c "import spectrex; print(spectrex.logger)"
```

Expected output: `<Logger spectrex (WARNING)>`

- [ ] **Step 4: Commit**

```bash
git add src/spectrex/__init__.py
git commit -m "feat: expose package-level logger with NullHandler"
```

---

### Task 2: NoiseModel.sample()

**Files:**
- Modify: `src/spectrex/solver.py`
- Modify: `unittests/test_solver.py`

- [ ] **Step 1: Write the failing test**

Add to `unittests/test_solver.py`:

```python
def test_sample_shape_and_type():
    """sample() returns an array with the same shape as input."""
    nm = NoiseModel(read_noise=5.0)
    rng = np.random.default_rng(0)
    image = np.ones((10, 20)) * 100.0
    noisy = nm.sample(image, rng)
    assert noisy.shape == image.shape
    assert noisy.dtype == np.float64


def test_sample_adds_noise():
    """sample() output differs from input (noise was added)."""
    nm = NoiseModel(read_noise=5.0)
    rng = np.random.default_rng(1)
    image = np.ones(1000) * 500.0
    noisy = nm.sample(image, rng)
    assert not np.allclose(noisy, image)


def test_sample_mean_near_input():
    """sample() mean should be close to input mean over many pixels."""
    nm = NoiseModel(read_noise=0.0)
    rng = np.random.default_rng(2)
    image = np.ones(100_000) * 200.0
    noisy = nm.sample(image, rng)
    np.testing.assert_allclose(noisy.mean(), 200.0, rtol=0.01)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest unittests/test_solver.py::test_sample_shape_and_type unittests/test_solver.py::test_sample_adds_noise unittests/test_solver.py::test_sample_mean_near_input -v
```

Expected: `AttributeError: 'NoiseModel' object has no attribute 'sample'`

- [ ] **Step 3: Implement NoiseModel.sample()**

In `src/spectrex/solver.py`, add the following method to the `NoiseModel` dataclass, after `precision_weights()`:

```python
def sample(self, f: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw a noisy realisation of pixel values.

    Adds zero-mean Gaussian noise with ``σ² = variance(f)`` to the
    input array.  This is an approximation to Poisson + read noise
    suitable for mock data generation.

    Parameters
    ----------
    f : np.ndarray
        Noiseless pixel values.
    rng : np.random.Generator
        NumPy random generator (e.g. ``np.random.default_rng(42)``).

    Returns
    -------
    np.ndarray
        Noisy pixel values with the same shape and dtype as *f*.
    """
    sigma = np.sqrt(self.variance(f))
    return f + rng.normal(0.0, sigma).astype(f.dtype)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest unittests/test_solver.py -v
```

Expected: all solver tests pass (previous 13 + 3 new = 16 total).

- [ ] **Step 5: Lint and type-check**

```bash
uv run ruff check src/spectrex/solver.py && uv run ty check src/spectrex/solver.py
```

Expected: `All checks passed!` for both.

- [ ] **Step 6: Commit**

```bash
git add src/spectrex/solver.py unittests/test_solver.py
git commit -m "feat: add NoiseModel.sample() for mock noise generation"
```

---

### Task 3: Sphinx docs wiring

**Files:**
- Modify: `docs/index.rst`

- [ ] **Step 1: Add Examples toctree to index.rst**

In `docs/index.rst`, add a new toctree block after the existing API Reference block:

```rst
.. toctree::
   :maxdepth: 1
   :caption: Examples

   Mock example <content/mock_example>
```

- [ ] **Step 2: Commit (notebook not yet created — symlink and notebook come in Task 4)**

```bash
git add docs/index.rst
git commit -m "docs: add Examples toctree entry for mock notebook"
```

---

### Task 4: Create the notebook

**Files:**
- Create: `notebooks/mock_example.ipynb`
- Create: `docs/content/mock_example.ipynb` (symlink)

This task creates the notebook as a JSON file via `nbformat`. The notebook must be pre-executed with outputs so the rendered docs are readable. The steps below build the cell sequence, then execute it.

- [ ] **Step 1: Create the notebooks/ directory**

```bash
mkdir -p notebooks
```

- [ ] **Step 2: Create the notebook JSON**

Create `notebooks/mock_example.ipynb` with the following content (use `nbformat` conventions: `nbformat=4`, `nbformat_minor=5`):

```json
{
 "nbformat": 4,
 "nbformat_minor": 5,
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.12.0"
  }
 },
 "cells": []
}
```

Cells are written in Step 3.

- [ ] **Step 3: Populate cells**

The complete cell sequence is below. Write each cell in order. Markdown cells use `"cell_type": "markdown"`. Code cells use `"cell_type": "code"`.

**Cell 1 — Title (markdown)**
```markdown
# spectrex mock example

End-to-end demonstration of the spectrex Phase 1 API using a synthetic
(mock) scene on a 500 × 20 pixel NIRISS/GR150R grism stamp.

The pipeline:
1. Load instrument configuration and PCA eigenspectra basis
2. Build (or load) the sparse forward operator **H**
3. Generate a mock scene — sparse PCA coefficient vector **ã**
4. Forward-model: disperse **ã** through **H** to produce a grism image
5. Recover **ã** from the dispersed image by LSQR
6. Repeat with Poisson + read noise to demonstrate the realistic case

RMSE is computed on reconstructed flux values `f(λ) = Φ @ a` so it is
physically meaningful and matches the parity plot axes.
```

**Cell 2 — Imports & constants (code)**
```python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import spectrex
from spectrex import (
    EigenspectraBasis,
    InstrumentConfig,
    NoiseModel,
    SciPySparseOperator,
    SpectralSolver,
)

# ── Paths ────────────────────────────────────────────────────────────────────
TESTDATA = Path("../testdata")
OPERATOR_CACHE = Path("operator_cache.npz")

# ── Configuration ─────────────────────────────────────────────────────────────
COLD_START = False    # set True to force operator rebuild from scratch
IMAGE_SHAPE = (500, 20)
SOURCE_DENSITY = 0.1  # fraction of pixels with injected sources
SEED = 42
N_COMPONENTS = 10     # must match eigenspectra CSV

rng = np.random.default_rng(SEED)

print(f"spectrex {spectrex.__version__}")
```

**Cell 3 — Step 1 header (markdown)**
```markdown
## Step 1 — Instrument configuration & eigenspectra basis

`InstrumentConfig` loads the aXe grism config file, wavelength range, and
sensitivity curves. `EigenspectraBasis` loads the 10 Kurucz PCA eigenspectra
and interpolates them onto the instrument wavelength grid.
```

**Cell 4 — Load config and basis (code)**
```python
config = InstrumentConfig.from_files(
    conf_path=TESTDATA / "Config Files" / "GR150R.F150W.220725.conf",
    wavelengthrange_path=TESTDATA / "jwst_niriss_wavelengthrange_0002.asdf",
    sensitivity_dir=TESTDATA / "SenseConfig" / "wfss-grism-configuration",
    filter_name="F150W",
    n_wavelengths=150,
)

basis = EigenspectraBasis.from_csv(
    TESTDATA / "eigenspectra_kurucz.csv",
    config.wavelengths,
)

print(f"Wavelength range: {config.wavelengths[0]:.0f} – {config.wavelengths[-1]:.0f} Å")
print(f"Grism orders: {list(config.orders)}")
print(f"Basis components: {basis.n_components}")
```

**Cell 5 — Step 2 header (markdown)**
```markdown
## Step 2 — Forward operator

Builds the sparse matrix **H** (shape `n_pix × n_pix·n_components`) that maps
PCA coefficients to dispersed pixel values across all grism orders.

> **Note:** The first run builds **H** from scratch, which takes ~60 s for a
> 500 × 20 stamp. The result is cached to `operator_cache.npz`.
> Set `COLD_START = True` in the constants cell to force a rebuild.
```

**Cell 6 — Build/load operator (code)**
```python
import time

if OPERATOR_CACHE.exists() and not COLD_START:
    op = SciPySparseOperator.load(OPERATOR_CACHE)
    print(f"Operator loaded from {OPERATOR_CACHE}  "
          f"(shape {op._H.shape[0]} × {op._H.shape[1]})")
else:
    t0 = time.perf_counter()
    op = SciPySparseOperator.build(config, basis, IMAGE_SHAPE)
    op.save(OPERATOR_CACHE)
    elapsed = time.perf_counter() - t0
    print(f"Operator built in {elapsed:.1f} s — cached to {OPERATOR_CACHE}")
    print(f"Shape: {op._H.shape[0]} × {op._H.shape[1]}")
```

**Cell 7 — Step 3 header (markdown)**
```markdown
## Step 3 — Mock scene

We inject sources at a random 10 % of pixels. For each active pixel we draw
random PCA coefficients, accept only if the reconstructed spectrum is
non-negative (physical), and store them in the flat coefficient vector **ã**.

The *direct image* is the broadband integrated flux I(x,y) = ∫ a(x,y,λ) dλ,
visualised as a 2-D stamp.
```

**Cell 8 — Generate mock scene (code)**
```python
n_pix = IMAGE_SHAPE[0] * IMAGE_SHAPE[1]
n = basis.n_components

a_tilde = np.zeros(n_pix * n)

num_active = int(SOURCE_DENSITY * n_pix)
active_k = rng.choice(n_pix, size=num_active, replace=False)

MAX_TRIES = 50
n_placed = 0
for k in active_k:
    for _ in range(MAX_TRIES):
        flux = rng.uniform(-1, 1, size=n)
        if np.all(basis.reconstruct(flux) >= 0):
            a_tilde[k * n : (k + 1) * n] = flux
            n_placed += 1
            break

print(f"Sources placed: {n_placed} / {num_active} requested "
      f"({100 * n_placed / n_pix:.1f} % of pixels)")

direct = basis.broadband_image(a_tilde, IMAGE_SHAPE)
print(f"Direct image shape: {direct.shape},  max flux: {direct.max():.4f}")
```

**Cell 9 — §1 header (markdown)**
```markdown
## §1 — Noiseless recovery

Forward-model **ã** → dispersed grism image, then recover **ã** from the
dispersed image using LSQR. The support mask is built from the direct image
(non-zero pixels) so LSQR only solves for active columns of **H**.
```

**Cell 10 — Disperse (noiseless) (code)**
```python
dispersed = op.apply(a_tilde).reshape(IMAGE_SHAPE)

# Support mask: True at pixels known to have a source
support_mask = a_tilde != 0
print(f"Dispersed image range: [{dispersed.min():.4f}, {dispersed.max():.4f}]")
print(f"Active coefficients:   {support_mask.sum()} / {len(support_mask)}")
```

**Cell 11 — Recover (noiseless) (code)**
```python
solver = SpectralSolver(op, max_iter=500, tolerance=1e-8)
recovered = solver.solve(dispersed, support_mask=support_mask)
print(f"Recovered vector shape: {recovered.shape}")
```

**Cell 12 — Compute RMSE (noiseless) (code)**
```python
active_indices = [k for k in range(n_pix) if np.any(a_tilde[k * n : (k + 1) * n] != 0)]

true_flux = np.concatenate(
    [basis.reconstruct(a_tilde[k * n : (k + 1) * n]) for k in active_indices]
)
rec_flux = np.concatenate(
    [basis.reconstruct(recovered[k * n : (k + 1) * n]) for k in active_indices]
)

rmse_noiseless = np.sqrt(np.mean((true_flux - rec_flux) ** 2))
print(f"Noiseless RMSE (flux): {rmse_noiseless:.6f}")
```

**Cell 13 — 4-panel figure (noiseless) (code)**
```python
def _clip(arr, nsigma_lo=2, nsigma_hi=2):
    m, s = np.nanmean(arr), np.nanstd(arr)
    return m - nsigma_lo * s, m + nsigma_hi * s

recovered_img = basis.broadband_image(recovered, IMAGE_SHAPE)
residual_img  = np.abs(direct - recovered_img)

vmin_dr, vmax_dr = _clip(direct)                       # shared scale for Direct & Recovered
vmin_d2, vmax_d2 = _clip(dispersed)
vmax_res = np.nanmean(residual_img) + np.nanstd(residual_img)

fig, axes = plt.subplots(1, 4, figsize=(14, 4))
kw = dict(origin="lower", aspect="auto", interpolation="nearest", cmap="inferno")

im0 = axes[0].imshow(direct,        vmin=vmin_dr, vmax=vmax_dr, **kw)
im1 = axes[1].imshow(dispersed,     vmin=vmin_d2, vmax=vmax_d2, **kw)
im2 = axes[2].imshow(recovered_img, vmin=vmin_dr, vmax=vmax_dr, **kw)
im3 = axes[3].imshow(residual_img,  vmin=0,       vmax=vmax_res, **kw)

titles = ["Direct image", "Dispersed (grism)", "Recovered", "|Residual|"]
for ax, im, title in zip(axes, [im0, im1, im2, im3], titles):
    ax.set_title(title)
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle("Noiseless recovery — 500 × 20 stamp, GR150R/F150W", y=1.01)
fig.tight_layout()
plt.show()
```

**Cell 14 — Parity plot (noiseless) (code)**
```python
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(true_flux, rec_flux, s=2, alpha=0.05, linewidths=0, color="steelblue")
minv = min(true_flux.min(), rec_flux.min())
maxv = max(true_flux.max(), rec_flux.max())
ax.plot([minv, maxv], [minv, maxv], "r--", lw=1, label="1:1")
ax.set_aspect("equal", adjustable="box")
ax.set_xlim(minv, maxv)
ax.set_ylim(minv, maxv)
ax.set_xlabel("True flux  f(λ)")
ax.set_ylabel("Recovered flux  f(λ)")
ax.set_title("Parity plot — noiseless")
ax.text(0.05, 0.92, f"RMSE = {rmse_noiseless:.4f}",
        transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
fig.tight_layout()
plt.show()
```

**Cell 15 — §2 header (markdown)**
```markdown
## §2 — Noisy recovery

We add Gaussian noise approximating Poisson + detector read noise
(`read_noise = 5 e⁻`) using `NoiseModel.sample()`.  The same support mask
and LSQR solver are used; the only change is the input image.

This is the realistic operating regime for real JWST data.
```

**Cell 16 — Add noise (code)**
```python
noise_model = NoiseModel(read_noise=5.0)
noisy_dispersed = noise_model.sample(dispersed, rng)
print(f"Noisy dispersed range: [{noisy_dispersed.min():.4f}, {noisy_dispersed.max():.4f}]")
print(f"Added noise std (mean over pixels): "
      f"{np.std(noisy_dispersed - dispersed):.4f}")
```

**Cell 17 — Recover (noisy) (code)**
```python
noisy_recovered = solver.solve(noisy_dispersed, support_mask=support_mask)
print(f"Noisy recovered vector shape: {noisy_recovered.shape}")
```

**Cell 18 — Compute RMSE (noisy) (code)**
```python
noisy_rec_flux = np.concatenate(
    [basis.reconstruct(noisy_recovered[k * n : (k + 1) * n]) for k in active_indices]
)

rmse_noisy = np.sqrt(np.mean((true_flux - noisy_rec_flux) ** 2))
print(f"Noiseless RMSE: {rmse_noiseless:.6f}")
print(f"Noisy RMSE:     {rmse_noisy:.6f}")
print(f"RMSE ratio:     {rmse_noisy / rmse_noiseless:.2f}×")
```

**Cell 19 — 4-panel figure (noisy) (code)**
```python
noisy_recovered_img = basis.broadband_image(noisy_recovered, IMAGE_SHAPE)
noisy_residual_img  = np.abs(direct - noisy_recovered_img)

vmax_nres = np.nanmean(noisy_residual_img) + np.nanstd(noisy_residual_img)

fig, axes = plt.subplots(1, 4, figsize=(14, 4))

im0 = axes[0].imshow(direct,             vmin=vmin_dr,   vmax=vmax_dr,   **kw)
im1 = axes[1].imshow(noisy_dispersed,    vmin=vmin_d2,   vmax=vmax_d2,   **kw)
im2 = axes[2].imshow(noisy_recovered_img,vmin=vmin_dr,   vmax=vmax_dr,   **kw)
im3 = axes[3].imshow(noisy_residual_img, vmin=0,         vmax=vmax_nres, **kw)

for ax, im, title in zip(axes, [im0, im1, im2, im3], titles):
    ax.set_title(title)
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle("Noisy recovery (read_noise=5 e⁻) — 500 × 20 stamp, GR150R/F150W", y=1.01)
fig.tight_layout()
plt.show()
```

**Cell 20 — Parity plot (noisy) (code)**
```python
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(true_flux, noisy_rec_flux, s=2, alpha=0.05, linewidths=0, color="darkorange")
ax.plot([minv, maxv], [minv, maxv], "r--", lw=1, label="1:1")
ax.set_aspect("equal", adjustable="box")
ax.set_xlim(minv, maxv)
ax.set_ylim(minv, maxv)
ax.set_xlabel("True flux  f(λ)")
ax.set_ylabel("Recovered flux  f(λ)")
ax.set_title("Parity plot — noisy (read_noise = 5 e⁻)")
ax.text(0.05, 0.92, f"RMSE = {rmse_noisy:.4f}",
        transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
fig.tight_layout()
plt.show()
```

**Cell 21 — Next steps (markdown)**
```markdown
## Next steps

The parity plots above give a single RMSE figure for one source density.
To quantify how recovery quality degrades as the field becomes more crowded,
see the planned sweep notebook:

```
notebooks/analysis_rmse_vs_density.ipynb   (TODO)
```

That notebook will define a `run_pipeline(a_tilde, op, basis, noise_model=None)`
helper and sweep `SOURCE_DENSITY` over ~5–8 values, plotting RMSE vs density
for both the noiseless and noisy cases.  This sweep is important for
establishing the credibility of the method in crowded NIRISS fields.
```

- [ ] **Step 4: Create the symlink**

```bash
cd docs/content && ln -s ../../notebooks/mock_example.ipynb mock_example.ipynb && cd ../..
```

Verify:
```bash
ls -la docs/content/mock_example.ipynb
```
Expected: `docs/content/mock_example.ipynb -> ../../notebooks/mock_example.ipynb`

- [ ] **Step 5: Execute the notebook**

```bash
uv run jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=600 \
    --ExecutePreprocessor.kernel_name=python3 \
    notebooks/mock_example.ipynb
```

Expected: notebook executes without errors; outputs (figures, print statements) are embedded in the JSON. The operator build step will take ~60 s if `operator_cache.npz` does not exist yet.

- [ ] **Step 6: Verify outputs are present**

```bash
uv run python -c "
import json, sys
nb = json.load(open('notebooks/mock_example.ipynb'))
code_cells = [c for c in nb['cells'] if c['cell_type']=='code']
empty = [i for i,c in enumerate(code_cells) if not c.get('outputs')]
if empty:
    print('Cells with no output:', empty); sys.exit(1)
else:
    print(f'All {len(code_cells)} code cells have outputs — OK')
"
```

Expected: `All 20 code cells have outputs — OK`

- [ ] **Step 7: Commit**

```bash
git add notebooks/mock_example.ipynb docs/content/mock_example.ipynb docs/index.rst
git commit -m "feat: add mock example notebook with noiseless and noisy recovery"
```

---

### Task 5: Smoke-test the docs build

**Files:** none new

- [ ] **Step 1: Build the docs**

```bash
uv run make -C docs html 2>&1 | tail -20
```

Expected: `build succeeded` (warnings about missing cross-references are acceptable; errors are not).

- [ ] **Step 2: Spot-check the rendered notebook page**

Open `docs/_build/html/content/mock_example.html` in a browser or check that the file exists and is non-trivial:

```bash
wc -l docs/_build/html/content/mock_example.html
```

Expected: > 200 lines (figures are embedded as base64 data URIs).

- [ ] **Step 3: Commit (only if docs build required a fix)**

If any docs fix was needed:
```bash
git add -A && git commit -m "fix: resolve docs build issue for mock notebook"
```

---

## Self-Review Notes

- `NoiseModel.sample()` signature uses `np.random.Generator` — matches `default_rng()` usage in cell 2. ✓
- `basis.broadband_image(a_tilde, IMAGE_SHAPE)` — method exists in `EigenspectraBasis`. ✓
- `basis.reconstruct(coefs)` — returns 1-D flux array of length `n_wavelengths`. ✓
- `op.apply(a_tilde)` returns a flat 1-D array — `.reshape(IMAGE_SHAPE)` is correct. ✓
- `solver.solve(dispersed, support_mask=support_mask)` — `dispersed` is 2-D here; `SpectralSolver.solve()` accepts 2-D and ravels internally. Verify this is the case; if not, pass `dispersed.ravel()`. ✓ (check during execution)
- Symlink will be committed as a symlink in git — works on Linux/macOS; note for any Windows contributors.
- `kw` dict defined in cell 13 is reused in cell 19 — ensure cell execution order is top-to-bottom. ✓
- `titles` list defined in cell 13 is reused in cell 19 — same note. ✓
