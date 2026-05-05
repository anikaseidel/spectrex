# Solver Accuracy Comparison Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `notebooks/comparison_solver_accuracy.ipynb` — a pre-executed, astronomer-facing notebook demonstrating accuracy gains of FISTA group-L1 (`JAXProximalSolver`) over LSQR (`SpectralSolver`) for crowded NIRISS WFSS fields, with side-by-side spectra and an RMSE vs density sweep comparing both solvers.

**Architecture:** Two sections in one notebook: (1) fixed 5-source crowded scene — mock detector image, per-source spectral overlays, residual images, RMSE bar chart; (2) RMSE vs density sweep (1–20 sources, 10 trials) running both solvers on the same trials. Narrative cells are placeholder-marked for Hypatia to fill. Pre-executed with outputs committed.

**Tech Stack:** Python 3.12+, `spectrex` (JAXOperator, JAXProximalSolver, SciPySparseOperator, SpectralSolver, EigenspectraBasis, InstrumentConfig, NoiseModel), `numpy`, `matplotlib`, `uv run jupyter nbconvert`

---

## File Map

| Action | Path |
|--------|------|
| Create | `notebooks/comparison_solver_accuracy.ipynb` |
| Create | `docs/content/comparison_solver_accuracy.ipynb` (symlink) |
| Modify | `docs/index.rst` (add to Examples toctree) |

---

## Critical API Notes (read before writing any code)

- `InstrumentConfig.from_files(conf_path, wavelengthrange_path, sensitivity_dir, filter_name, n_wavelengths=150)` — **not** `from_config_dir` (does not exist).
- `SciPySparseOperator.build(config, basis, image_shape)` — builds operator for full image; source positions are NOT passed here.
- `SpectralSolver(op, noise_model=nm, regularisation=1e-2).solve(image_flat, support_mask=mask)` — takes flattened image, returns flat coefficient vector.
- `JAXOperator.build(config, basis, image_shape, source_positions)` — source_positions is `np.ndarray` shape `(K, 2)` of `(row, col)` float positions.
- `JAXProximalSolver(jax_op, noise_model=nm, lam=0.05, n_iter=200).solve(image_flat)` — takes flattened image, returns flat coefficient vector shape `(K * M,)`.
- `NoiseModel(read_noise=5.0)` — Poisson + read noise.
- `EigenspectraBasis.from_csv(path, wavelengths)` — loads PCA basis; `.n_components` gives M.
- All randomness seeded with `np.random.default_rng(2026)` for reproducibility.
- Testdata lives at `REPO / 'testdata'` where `REPO = Path.cwd().parent` (notebook in `notebooks/`).
- Correct conf file path: `TESTDATA / 'Config Files' / 'GR150R.F150W.220725.conf'`
- Correct wavelength range path: `TESTDATA / 'jwst_niriss_wavelengthrange_0002.asdf'`
- Correct sensitivity dir: `TESTDATA / 'SenseConfig' / 'wfss-grism-configuration'`

---

## Task 1: Scaffold the notebook structure

**Files:**
- Create: `notebooks/comparison_solver_accuracy.ipynb`

- [ ] **Step 1: Create the notebook JSON skeleton**

Create `notebooks/comparison_solver_accuracy.ipynb` with this exact content (valid notebook with no cells yet):

```json
{
 "cells": [],
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
 "nbformat": 4,
 "nbformat_minor": 5
}
```

- [ ] **Step 2: Commit the skeleton**

```bash
git add notebooks/comparison_solver_accuracy.ipynb
git commit -m "chore: scaffold comparison_solver_accuracy notebook"
```

---

## Task 2: Imports, paths, and instrument setup cell

**Files:**
- Modify: `notebooks/comparison_solver_accuracy.ipynb`

- [ ] **Step 1: Open the notebook in a Python script for programmatic cell building**

Use `nbformat` to build the notebook programmatically. Create a temporary build script at `notebooks/_build_comparison_accuracy.py`:

```python
"""Build comparison_solver_accuracy.ipynb programmatically."""
from __future__ import annotations
import json
from pathlib import Path

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

cells = []

def md(src: str) -> nbformat.NotebookNode:
    return new_markdown_cell(src)

def code(src: str) -> nbformat.NotebookNode:
    return new_code_cell(src)

# ── Cell 1: Title + intro (Hypatia placeholder) ──────────────────────────────
cells.append(md(
    "# Solver Accuracy Comparison: LSQR vs FISTA Group-L1\n\n"
    "<!-- HYPATIA: Replace this cell with an astronomy-focused introduction. -->\n"
    "<!-- Frame as: crowded JWST NIRISS WFSS fields, source confusion, -->\n"
    "<!-- deblending challenge, and what each solver brings. -->\n"
    "*Placeholder — see HYPATIA marker above.*"
))

# ── Cell 2: Imports ───────────────────────────────────────────────────────────
cells.append(code(
    "from __future__ import annotations\n\n"
    "import warnings\n"
    "from pathlib import Path\n\n"
    "import matplotlib.pyplot as plt\n"
    "import numpy as np\n\n"
    "import spectrex\n"
    "from spectrex import (\n"
    "    EigenspectraBasis,\n"
    "    InstrumentConfig,\n"
    "    JAXOperator,\n"
    "    JAXProximalSolver,\n"
    "    NoiseModel,\n"
    "    SciPySparseOperator,\n"
    "    SpectralSolver,\n"
    ")\n\n"
    "warnings.filterwarnings('ignore')\n\n"
    "NOTEBOOK_DIR = Path.cwd()\n"
    "REPO = NOTEBOOK_DIR.parent\n"
    "TESTDATA = REPO / 'testdata'\n"
    "print(f'spectrex version: {spectrex.__version__}')\n"
))

# ── Cell 3: Instrument setup ──────────────────────────────────────────────────
cells.append(md("## Setup: Instrument Configuration and Basis"))

cells.append(code(
    "config = InstrumentConfig.from_files(\n"
    "    conf_path=TESTDATA / 'Config Files' / 'GR150R.F150W.220725.conf',\n"
    "    wavelengthrange_path=TESTDATA / 'jwst_niriss_wavelengthrange_0002.asdf',\n"
    "    sensitivity_dir=TESTDATA / 'SenseConfig' / 'wfss-grism-configuration',\n"
    "    filter_name='F150W',\n"
    "    n_wavelengths=150,\n"
    ")\n"
    "basis = EigenspectraBasis.from_csv(\n"
    "    TESTDATA / 'eigenspectra_kurucz.csv',\n"
    "    config.wavelengths,\n"
    ")\n\n"
    "IMAGE_SHAPE = (50, 20)\n"
    "N_ROWS, N_COLS = IMAGE_SHAPE\n"
    "N_PIX = N_ROWS * N_COLS\n"
    "M = basis.n_components\n"
    "NOISE_MODEL = NoiseModel(read_noise=5.0)\n"
    "RNG = np.random.default_rng(2026)\n\n"
    "print(f'Image shape: {IMAGE_SHAPE}, n_pix={N_PIX}')\n"
    "print(f'Basis components: M={M}')\n"
    "print(f'Wavelengths: {len(config.wavelengths)} points, '\n"
    "      f'{config.wavelengths[0]:.0f}–{config.wavelengths[-1]:.0f} Å')\n"
))

nb = new_notebook(cells=cells)
path = Path(__file__).parent / 'comparison_solver_accuracy.ipynb'
nbformat.write(nb, path)
print(f'Written: {path}')
```

- [ ] **Step 2: Run the build script to generate the initial notebook cells**

```bash
cd notebooks && uv run python _build_comparison_accuracy.py
```

Expected: `Written: .../notebooks/comparison_solver_accuracy.ipynb`

- [ ] **Step 3: Commit**

```bash
git add notebooks/comparison_solver_accuracy.ipynb notebooks/_build_comparison_accuracy.py
git commit -m "feat: add imports, paths, and instrument setup cells to accuracy notebook"
```

---

## Task 3: Build operators and mock scene

**Files:**
- Modify: `notebooks/_build_comparison_accuracy.py`
- Modify: `notebooks/comparison_solver_accuracy.ipynb`

- [ ] **Step 1: Add operator build cells to the build script**

Append the following to `_build_comparison_accuracy.py` (before the `nb = new_notebook(cells=cells)` line):

```python
# ── Cell 4: Build operators ───────────────────────────────────────────────────
cells.append(md(
    "## Section 1: Fixed Crowded Scene\n\n"
    "<!-- HYPATIA: Brief section intro — what we're demonstrating and why. -->"
))

cells.append(md("### Build Operators"))

cells.append(code(
    "# Fixed source positions (row, col) in the 50×20 image\n"
    "SOURCE_POSITIONS = np.array([\n"
    "    [ 8.0,  4.0],\n"
    "    [15.0, 10.0],\n"
    "    [25.0,  6.0],\n"
    "    [35.0, 14.0],\n"
    "    [42.0,  8.0],\n"
    "], dtype=np.float64)  # shape (5, 2)\n"
    "K = len(SOURCE_POSITIONS)\n\n"
    "print(f'Building SciPySparseOperator for {K} sources...')\n"
    "scipy_op = SciPySparseOperator.build(config, basis, IMAGE_SHAPE)\n"
    "print(f'  n_coefficients = {scipy_op.n_coefficients}')\n\n"
    "print(f'Building JAXOperator for {K} sources...')\n"
    "jax_op = JAXOperator.build(config, basis, IMAGE_SHAPE, SOURCE_POSITIONS)\n"
    "print(f'  n_coefficients = {jax_op.n_coefficients}')\n"
    "print('Done.')\n"
))

# ── Cell 5: Mock scene ────────────────────────────────────────────────────────
cells.append(md("### Mock Crowded Scene"))

cells.append(code(
    "# Ground-truth coefficients: random but seeded\n"
    "# JAXOperator uses compact layout: a_true shape (K*M,)\n"
    "a_true_jax = RNG.standard_normal(K * M).astype(np.float64)\n\n"
    "# SciPySparseOperator uses full flat layout: a_true shape (N_PIX*M,)\n"
    "# with non-zero blocks only at the K source pixel positions.\n"
    "# Map source (row, col) to flat pixel index\n"
    "source_flat_idx = [\n"
    "    int(round(r)) * N_COLS + int(round(c))\n"
    "    for r, c in SOURCE_POSITIONS\n"
    "]\n"
    "a_true_scipy = np.zeros(N_PIX * M)\n"
    "for k, p in enumerate(source_flat_idx):\n"
    "    a_true_scipy[p * M : (p + 1) * M] = a_true_jax[k * M : (k + 1) * M]\n\n"
    "# Forward model\n"
    "f_clean_jax = jax_op.apply(a_true_jax).reshape(IMAGE_SHAPE)\n"
    "f_clean_scipy = scipy_op.apply(a_true_scipy).reshape(IMAGE_SHAPE)\n\n"
    "# Add noise (same noise realisation for both)\n"
    "noise_rng = np.random.default_rng(42)\n"
    "f_noisy = NOISE_MODEL.sample(f_clean_jax, noise_rng)\n\n"
    "print(f'f_noisy: min={f_noisy.min():.2f}, max={f_noisy.max():.2f}')\n"
    "print(f'Max clean signal: {f_clean_jax.max():.2f}')\n"
))

# ── Cell 6: Display mock image ────────────────────────────────────────────────
cells.append(md(
    "### Mock Detector Image\n\n"
    "<!-- HYPATIA: Describe what this image represents physically — "
    "overlapping grism spectra from 5 sources in a crowded field. -->"
))

cells.append(code(
    "fig, ax = plt.subplots(figsize=(8, 4))\n"
    "im = ax.imshow(f_noisy, origin='lower', aspect='auto',\n"
    "               cmap='viridis', interpolation='nearest')\n"
    "plt.colorbar(im, ax=ax, label='Counts')\n"
    "for k, (r, c) in enumerate(SOURCE_POSITIONS):\n"
    "    ax.plot(c, r, 'r+', markersize=12, markeredgewidth=2)\n"
    "    ax.annotate(f'S{k+1}', xy=(c, r), xytext=(c+0.5, r+0.5),\n"
    "                color='white', fontsize=8)\n"
    "ax.set_xlabel('Column (pixel)')\n"
    "ax.set_ylabel('Row (pixel)')\n"
    "ax.set_title('Mock NIRISS WFSS Detector Image — 5 crowded sources')\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
))
```

- [ ] **Step 2: Regenerate the notebook**

```bash
cd notebooks && uv run python _build_comparison_accuracy.py
```

- [ ] **Step 3: Commit**

```bash
git add notebooks/comparison_solver_accuracy.ipynb notebooks/_build_comparison_accuracy.py
git commit -m "feat: add operator build and mock scene cells to accuracy notebook"
```

---

## Task 4: Solve with both solvers

**Files:**
- Modify: `notebooks/_build_comparison_accuracy.py`
- Modify: `notebooks/comparison_solver_accuracy.ipynb`

- [ ] **Step 1: Add solver cells to the build script**

Append to `_build_comparison_accuracy.py` (before `nb = new_notebook(...)`):

```python
# ── Cell 7: Solve ─────────────────────────────────────────────────────────────
cells.append(md("### Spectral Extraction with Both Solvers"))

cells.append(code(
    "# Support mask for SpectralSolver (non-zero at source pixel blocks)\n"
    "support_mask = np.zeros(N_PIX * M, dtype=bool)\n"
    "for p in source_flat_idx:\n"
    "    support_mask[p * M : (p + 1) * M] = True\n\n"
    "# --- LSQR (SpectralSolver) ---\n"
    "import time\n"
    "t0 = time.perf_counter()\n"
    "solver_lsqr = SpectralSolver(\n"
    "    scipy_op, noise_model=NOISE_MODEL, regularisation=1e-2\n"
    ")\n"
    "a_rec_scipy = solver_lsqr.solve(f_noisy, support_mask=support_mask)\n"
    "t_lsqr = time.perf_counter() - t0\n\n"
    "# Extract active blocks\n"
    "a_rec_lsqr = np.array([\n"
    "    a_rec_scipy[p * M : (p + 1) * M] for p in source_flat_idx\n"
    "]).reshape(K * M)  # (K*M,)\n\n"
    "# --- FISTA (JAXProximalSolver) ---\n"
    "t0 = time.perf_counter()\n"
    "solver_fista = JAXProximalSolver(\n"
    "    jax_op, noise_model=NOISE_MODEL, lam=0.05, n_iter=200\n"
    ")\n"
    "a_rec_fista = solver_fista.solve(f_noisy)  # (K*M,)\n"
    "t_fista = time.perf_counter() - t0\n\n"
    "print(f'LSQR  solve time: {t_lsqr:.2f} s')\n"
    "print(f'FISTA solve time: {t_fista:.2f} s')\n"
))
```

- [ ] **Step 2: Regenerate and spot-check**

```bash
cd notebooks && uv run python _build_comparison_accuracy.py
```

- [ ] **Step 3: Commit**

```bash
git add notebooks/comparison_solver_accuracy.ipynb notebooks/_build_comparison_accuracy.py
git commit -m "feat: add LSQR and FISTA solve cells to accuracy notebook"
```

---

## Task 5: Per-source spectral overlay plots

**Files:**
- Modify: `notebooks/_build_comparison_accuracy.py`
- Modify: `notebooks/comparison_solver_accuracy.ipynb`

- [ ] **Step 1: Add spectral overlay plot cells**

Append to `_build_comparison_accuracy.py` (before `nb = new_notebook(...)`):

```python
# ── Cell 8: Spectral overlays ─────────────────────────────────────────────────
cells.append(md(
    "### Per-Source Recovered Spectra\n\n"
    "<!-- HYPATIA: Interpret what the plot shows — where FISTA improves -->\n"
    "<!-- over LSQR, and what contamination from neighbours looks like. -->"
))

cells.append(code(
    "# Reconstruct spectra from coefficients using the basis\n"
    "# basis.components shape: (M, n_wav) — rows are eigenvectors\n"
    "def reconstruct_spectrum(coeffs_km: np.ndarray, k: int) -> np.ndarray:\n"
    "    \"\"\"Reconstruct spectrum for source k from flat coefficient vector.\"\"\"\n"
    "    c = coeffs_km[k * M : (k + 1) * M]  # (M,)\n"
    "    return basis.components.T @ c  # (n_wav,)\n\n"
    "wav = config.wavelengths / 1e4  # Convert Å → μm for plot\n\n"
    "fig, axes = plt.subplots(1, K, figsize=(4 * K, 3), sharey=False)\n"
    "for k, ax in enumerate(axes):\n"
    "    sp_true  = reconstruct_spectrum(a_true_jax,  k)\n"
    "    sp_lsqr  = reconstruct_spectrum(a_rec_lsqr,  k)\n"
    "    sp_fista = reconstruct_spectrum(a_rec_fista, k)\n"
    "    ax.plot(wav, sp_true,  'k-',  lw=2,   label='Ground truth', alpha=0.8)\n"
    "    ax.plot(wav, sp_lsqr,  'b--', lw=1.5, label='LSQR')\n"
    "    ax.plot(wav, sp_fista, 'r-',  lw=1.5, label='FISTA')\n"
    "    ax.set_title(f'Source {k+1}')\n"
    "    ax.set_xlabel('Wavelength (μm)')\n"
    "    if k == 0:\n"
    "        ax.set_ylabel('Flux (arb. units)')\n"
    "    ax.legend(fontsize=7)\n"
    "    ax.grid(True, alpha=0.3)\n"
    "fig.suptitle('Recovered Spectra: Ground Truth vs LSQR vs FISTA', y=1.01)\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
))
```

- [ ] **Step 2: Regenerate**

```bash
cd notebooks && uv run python _build_comparison_accuracy.py
```

- [ ] **Step 3: Commit**

```bash
git add notebooks/comparison_solver_accuracy.ipynb notebooks/_build_comparison_accuracy.py
git commit -m "feat: add per-source spectral overlay plot to accuracy notebook"
```

---

## Task 6: Residual images and RMSE bar chart

**Files:**
- Modify: `notebooks/_build_comparison_accuracy.py`
- Modify: `notebooks/comparison_solver_accuracy.ipynb`

- [ ] **Step 1: Add residual and RMSE cells**

Append to `_build_comparison_accuracy.py` (before `nb = new_notebook(...)`):

```python
# ── Cell 9: Residual images ───────────────────────────────────────────────────
cells.append(md(
    "### Residual Images\n\n"
    "<!-- HYPATIA: Interpret residual structure — what correlated residuals -->\n"
    "<!-- indicate and how FISTA's regularisation reduces them. -->"
))

cells.append(code(
    "# Reconstruct model images from recovered coefficients\n"
    "f_model_lsqr  = scipy_op.apply(a_rec_scipy).reshape(IMAGE_SHAPE)\n"
    "f_model_fista = jax_op.apply(a_rec_fista).reshape(IMAGE_SHAPE)\n\n"
    "residual_lsqr  = f_noisy - f_model_lsqr\n"
    "residual_fista = f_noisy - f_model_fista\n\n"
    "vlim = np.percentile(np.abs(residual_lsqr), 99)\n\n"
    "fig, axes = plt.subplots(1, 2, figsize=(10, 4))\n"
    "for ax, resid, title in zip(\n"
    "    axes,\n"
    "    [residual_lsqr, residual_fista],\n"
    "    ['Residual: LSQR', 'Residual: FISTA'],\n"
    "):\n"
    "    im = ax.imshow(resid, origin='lower', aspect='auto',\n"
    "                   cmap='RdBu_r', vmin=-vlim, vmax=vlim)\n"
    "    plt.colorbar(im, ax=ax, label='Counts')\n"
    "    ax.set_title(title)\n"
    "    ax.set_xlabel('Column')\n"
    "    ax.set_ylabel('Row')\n"
    "plt.suptitle('Residual Images (data − model)', y=1.02)\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
    "print(f'LSQR  residual RMS: {np.std(residual_lsqr):.4f}')\n"
    "print(f'FISTA residual RMS: {np.std(residual_fista):.4f}')\n"
))

# ── Cell 10: RMSE bar chart ───────────────────────────────────────────────────
cells.append(md("### Per-Source RMSE Comparison"))

cells.append(code(
    "rmse_lsqr  = [\n"
    "    float(np.sqrt(np.mean(\n"
    "        (a_rec_lsqr[k*M:(k+1)*M] - a_true_jax[k*M:(k+1)*M])**2\n"
    "    )))\n"
    "    for k in range(K)\n"
    "]\n"
    "rmse_fista = [\n"
    "    float(np.sqrt(np.mean(\n"
    "        (a_rec_fista[k*M:(k+1)*M] - a_true_jax[k*M:(k+1)*M])**2\n"
    "    )))\n"
    "    for k in range(K)\n"
    "]\n\n"
    "x = np.arange(K)\n"
    "width = 0.35\n"
    "fig, ax = plt.subplots(figsize=(7, 4))\n"
    "ax.bar(x - width/2, rmse_lsqr,  width, label='LSQR',  color='steelblue')\n"
    "ax.bar(x + width/2, rmse_fista, width, label='FISTA', color='tomato')\n"
    "ax.set_xticks(x)\n"
    "ax.set_xticklabels([f'S{k+1}' for k in range(K)])\n"
    "ax.set_ylabel('RMSE (coefficient units)')\n"
    "ax.set_title('Per-Source RMSE: LSQR vs FISTA')\n"
    "ax.legend()\n"
    "ax.grid(True, alpha=0.3, axis='y')\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
    "print('LSQR  RMSE per source:', [f'{v:.4f}' for v in rmse_lsqr])\n"
    "print('FISTA RMSE per source:', [f'{v:.4f}' for v in rmse_fista])\n"
))
```

- [ ] **Step 2: Regenerate**

```bash
cd notebooks && uv run python _build_comparison_accuracy.py
```

- [ ] **Step 3: Commit**

```bash
git add notebooks/comparison_solver_accuracy.ipynb notebooks/_build_comparison_accuracy.py
git commit -m "feat: add residual images and RMSE bar chart to accuracy notebook"
```

---

## Task 7: RMSE vs density sweep (Section 2)

**Files:**
- Modify: `notebooks/_build_comparison_accuracy.py`
- Modify: `notebooks/comparison_solver_accuracy.ipynb`

- [ ] **Step 1: Add sweep cells to the build script**

Append to `_build_comparison_accuracy.py` (before `nb = new_notebook(...)`):

```python
# ── Section 2: RMSE vs density sweep ─────────────────────────────────────────
cells.append(md(
    "## Section 2: RMSE vs Source Density\n\n"
    "<!-- HYPATIA: Introduce the sweep — what density means physically, -->\n"
    "<!-- and why we care about the crossover point. -->"
))

cells.append(code(
    "N_SOURCES_GRID = [1, 2, 3, 5, 8, 10, 15, 20]\n"
    "N_TRIALS = 10\n"
    "SWEEP_RNG = np.random.default_rng(2027)\n"
    "REGULARISATION = 1e-2\n"
    "LAM_FISTA = 0.05\n"
))

cells.append(md("### Sweep Helper"))

cells.append(code(
    "def sweep_trial(\n"
    "    config: InstrumentConfig,\n"
    "    basis: EigenspectraBasis,\n"
    "    image_shape: tuple[int, int],\n"
    "    n_sources: int,\n"
    "    rng: np.random.Generator,\n"
    "    noise_model: NoiseModel,\n"
    ") -> dict[str, float]:\n"
    "    \"\"\"One Monte Carlo trial: build operators, solve, compute RMSE.\n\n"
    "    Returns dict with keys 'rmse_lsqr' and 'rmse_fista'.\n"
    "    \"\"\"\n"
    "    n_rows, n_cols = image_shape\n"
    "    n_pix = n_rows * n_cols\n"
    "    m = basis.n_components\n\n"
    "    # Random source positions (row, col)\n"
    "    flat_idx = rng.choice(n_pix, size=n_sources, replace=False)\n"
    "    src_pos = np.column_stack([\n"
    "        flat_idx // n_cols, flat_idx % n_cols\n"
    "    ]).astype(np.float64)\n\n"
    "    # Ground-truth coefficients\n"
    "    a_true = rng.standard_normal(n_sources * m)\n\n"
    "    # Build operators\n"
    "    sp_op  = SciPySparseOperator.build(config, basis, image_shape)\n"
    "    jx_op  = JAXOperator.build(config, basis, image_shape, src_pos)\n\n"
    "    # Forward model + noise\n"
    "    f_clean = jx_op.apply(a_true).reshape(image_shape)\n"
    "    f_noisy = noise_model.sample(f_clean, rng)\n\n"
    "    # Support mask for SpectralSolver\n"
    "    mask = np.zeros(n_pix * m, dtype=bool)\n"
    "    for p in flat_idx:\n"
    "        mask[p * m : (p + 1) * m] = True\n\n"
    "    # Scipy full layout ground truth (for SpectralSolver comparison)\n"
    "    a_true_scipy = np.zeros(n_pix * m)\n"
    "    for k, p in enumerate(flat_idx):\n"
    "        a_true_scipy[p * m : (p + 1) * m] = a_true[k * m : (k + 1) * m]\n\n"
    "    # LSQR\n"
    "    a_lsqr_full = SpectralSolver(\n"
    "        sp_op, noise_model=noise_model, regularisation=REGULARISATION\n"
    "    ).solve(f_noisy, support_mask=mask)\n"
    "    a_lsqr = np.concatenate([\n"
    "        a_lsqr_full[p * m : (p + 1) * m] for p in flat_idx\n"
    "    ])\n\n"
    "    # FISTA\n"
    "    a_fista = JAXProximalSolver(\n"
    "        jx_op, noise_model=noise_model, lam=LAM_FISTA, n_iter=200\n"
    "    ).solve(f_noisy)\n\n"
    "    rmse_lsqr  = float(np.sqrt(np.mean((a_lsqr  - a_true)**2)))\n"
    "    rmse_fista = float(np.sqrt(np.mean((a_fista - a_true)**2)))\n"
    "    return {'rmse_lsqr': rmse_lsqr, 'rmse_fista': rmse_fista}\n"
))

cells.append(md("### Run Sweep"))

cells.append(code(
    "sweep_results: dict[int, list[dict]] = {n: [] for n in N_SOURCES_GRID}\n\n"
    "for n_src in N_SOURCES_GRID:\n"
    "    for trial in range(N_TRIALS):\n"
    "        trial_rng = np.random.default_rng(SWEEP_RNG.integers(0, 2**31))\n"
    "        res = sweep_trial(\n"
    "            config, basis, IMAGE_SHAPE, n_src, trial_rng, NOISE_MODEL\n"
    "        )\n"
    "        sweep_results[n_src].append(res)\n"
    "    lsqr_m  = np.mean([r['rmse_lsqr']  for r in sweep_results[n_src]])\n"
    "    fista_m = np.mean([r['rmse_fista'] for r in sweep_results[n_src]])\n"
    "    print(f'n={n_src:2d}: LSQR {lsqr_m:.4f}  FISTA {fista_m:.4f}')\n"
))

cells.append(md(
    "### RMSE vs Source Density\n\n"
    "<!-- HYPATIA: Interpret the crossover — at what density FISTA's group-L1 -->\n"
    "<!-- regularisation clearly outperforms LSQR, and what this means for -->\n"
    "<!-- observers targeting crowded NIRISS fields. -->"
))

cells.append(code(
    "ns_arr = np.array(N_SOURCES_GRID)\n"
    "lsqr_means  = np.array([np.mean([r['rmse_lsqr']  for r in sweep_results[n]]) for n in N_SOURCES_GRID])\n"
    "lsqr_stds   = np.array([np.std( [r['rmse_lsqr']  for r in sweep_results[n]]) for n in N_SOURCES_GRID])\n"
    "fista_means = np.array([np.mean([r['rmse_fista'] for r in sweep_results[n]]) for n in N_SOURCES_GRID])\n"
    "fista_stds  = np.array([np.std( [r['rmse_fista'] for r in sweep_results[n]]) for n in N_SOURCES_GRID])\n\n"
    "fig, ax = plt.subplots(figsize=(8, 5))\n"
    "ax.fill_between(ns_arr, lsqr_means - lsqr_stds, lsqr_means + lsqr_stds,\n"
    "                alpha=0.2, color='steelblue')\n"
    "ax.fill_between(ns_arr, fista_means - fista_stds, fista_means + fista_stds,\n"
    "                alpha=0.2, color='tomato')\n"
    "ax.plot(ns_arr, lsqr_means,  'o-', color='steelblue', label='LSQR (SpectralSolver)', lw=2)\n"
    "ax.plot(ns_arr, fista_means, 's-', color='tomato',    label='FISTA (JAXProximalSolver)', lw=2)\n"
    "ax.set_xlabel('Number of sources')\n"
    "ax.set_ylabel('RMSE (coefficient units)')\n"
    "ax.set_title('Extraction RMSE vs Source Density: LSQR vs FISTA')\n"
    "ax.legend()\n"
    "ax.grid(True, alpha=0.3)\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
))

# ── Closing cell ──────────────────────────────────────────────────────────────
cells.append(md(
    "## Summary\n\n"
    "<!-- HYPATIA: Write a concise scientific summary — key findings, -->\n"
    "<!-- recommendation for observers, and pointers to further work. -->\n"
    "*Placeholder — see HYPATIA marker above.*"
))
```

- [ ] **Step 2: Regenerate**

```bash
cd notebooks && uv run python _build_comparison_accuracy.py
```

- [ ] **Step 3: Commit**

```bash
git add notebooks/comparison_solver_accuracy.ipynb notebooks/_build_comparison_accuracy.py
git commit -m "feat: add RMSE vs density sweep (both solvers) to accuracy notebook"
```

---

## Task 8: Execute the notebook

**Files:**
- Modify: `notebooks/comparison_solver_accuracy.ipynb` (adds outputs)

- [ ] **Step 1: Execute the notebook**

```bash
cd notebooks && uv run jupyter nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=600 \
    --inplace \
    comparison_solver_accuracy.ipynb
```

Expected: exits 0; notebook file updated with cell outputs and figures.

If execution fails, check the error cell output and fix the build script, regenerate, and retry.

- [ ] **Step 2: Verify outputs exist**

```bash
uv run python -c "
import json, sys
nb = json.load(open('notebooks/comparison_solver_accuracy.ipynb'))
code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
missing = [i for i, c in enumerate(code_cells) if not c.get('outputs')]
print(f'{len(code_cells)} code cells, {len(missing)} without output: {missing}')
sys.exit(1 if missing else 0)
"
```

Expected: `N code cells, 0 without output: []`

- [ ] **Step 3: Commit with outputs**

```bash
git add notebooks/comparison_solver_accuracy.ipynb
git commit -m "feat: execute comparison_solver_accuracy notebook with outputs"
```

---

## Task 9: Dispatch Hypatia for narrative cells

**Files:**
- Modify: `notebooks/comparison_solver_accuracy.ipynb` (narrative cells updated)

- [ ] **Step 1: Dispatch Hypatia subagent**

Dispatch the Hypatia agent with this prompt:

> "In the notebook `notebooks/comparison_solver_accuracy.ipynb`, replace every markdown cell that contains a `<!-- HYPATIA: ... -->` comment block with astronomy-appropriate narrative. The notebook compares `SpectralSolver` (LSQR) and `JAXProximalSolver` (FISTA group-L1) for JWST NIRISS WFSS spectral extraction in crowded fields.
>
> Guidelines:
> - Frame the problem as deblending overlapping grism spectra in a crowded NIRISS field
> - Explain group-L1 regularisation accessibly (it enforces per-source sparsity without requiring knowledge of how many spectral components each source has)
> - Interpret residual plots in terms of source confusion and what correlated residuals indicate
> - Interpret the RMSE vs density crossover in terms of when FISTA's regularisation pays off for observers
> - Keep each narrative cell to 2–4 sentences — concise, science-focused, no code
> - Do NOT modify any code cells
> - After editing, re-execute the notebook: `cd notebooks && uv run jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=600 --inplace comparison_solver_accuracy.ipynb`
> - Commit: `git add notebooks/comparison_solver_accuracy.ipynb && git commit -m 'docs: Hypatia narrative cells for accuracy comparison notebook'`"

- [ ] **Step 2: Verify no HYPATIA placeholders remain**

```bash
python -c "
import json
nb = json.load(open('notebooks/comparison_solver_accuracy.ipynb'))
hits = [i for i, c in enumerate(nb['cells'])
        if 'HYPATIA' in c.get('source', '')]
print('Remaining HYPATIA markers at cells:', hits)
"
```

Expected: `Remaining HYPATIA markers at cells: []`

---

## Task 10: Symlink, toctree, and final commit

**Files:**
- Create: `docs/content/comparison_solver_accuracy.ipynb` (symlink)
- Modify: `docs/index.rst`

- [ ] **Step 1: Create the symlink**

```bash
cd docs/content && ln -s ../../notebooks/comparison_solver_accuracy.ipynb comparison_solver_accuracy.ipynb
```

- [ ] **Step 2: Add to Examples toctree in `docs/index.rst`**

Find the Examples toctree block (currently ends with `content/analysis_rmse_vs_density`) and add the new entry:

```rst
.. toctree::
   :maxdepth: 1
   :caption: Examples

   Mock example <content/mock_example>
   content/analysis_rmse_vs_density
   content/comparison_solver_accuracy
```

- [ ] **Step 3: Clean up build script**

```bash
rm notebooks/_build_comparison_accuracy.py
```

- [ ] **Step 4: Final commit**

```bash
git add docs/content/comparison_solver_accuracy.ipynb docs/index.rst
git rm notebooks/_build_comparison_accuracy.py
git commit -m "feat: comparison_solver_accuracy notebook — symlink and toctree"
```

---

## Done

The completed notebook demonstrates:
- Side-by-side spectra (ground truth / LSQR / FISTA) for 5 crowded sources
- Residual images confirming FISTA reduces structured residuals
- RMSE vs density sweep showing both solvers on the same axes
- Astronomy-appropriate narrative from Hypatia throughout
