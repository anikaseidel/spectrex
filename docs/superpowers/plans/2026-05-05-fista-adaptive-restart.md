# FISTA Adaptive Restart Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add gradient-based adaptive restart, convergence tolerance, and a per-iteration callback to `JAXProximalSolver`; fix the mismatched residual metric in `comparison_computational.ipynb`.

**Architecture:** All changes are confined to `jax_solver.py` (solver logic), `test_jax_solver.py` (three new tests), and `comparison_computational.ipynb` (cells 20 and 22, plus one new markdown cell). No public API breakage — new parameters are keyword-only with backward-compatible defaults. `restart=True` is the new default; setting `restart=False` exactly reproduces the old behaviour.

**Tech Stack:** Python 3.12, JAX 0.10, NumPy ≥ 2, pytest, uv

---

## Files

| File | Change |
|------|--------|
| `src/spectrex/jax_solver.py` | Add `tol`, `restart`, `callback` to `__init__` + `_solve()` loop changes + updated docstring |
| `unittests/test_jax_solver.py` | 3 new tests appended |
| `notebooks/comparison_computational.ipynb` | Replace cell 20; insert markdown cell; update cell 22 axis label |

---

## Task 1: Update `JAXProximalSolver.__init__` and class docstring

**Files:**
- Modify: `src/spectrex/jax_solver.py:96-138`

- [ ] **Step 1: Replace the `__init__` signature and body**

Replace everything from the class docstring through `self._step: float | None = None` with:

```python
class JAXProximalSolver:
    """FISTA proximal gradient solver with group-L1 regularisation.

    Minimises::

        (1/2) ||W (H a - f)||² + λ Σ_k ||a_k||₂

    where ``W = diag(precision_weights)`` and the group-L1 penalty
    zeros entire source groups (index *k* over basis components *m*).

    The Lipschitz constant *L* of the gradient is estimated once at
    construction via power iteration; step size is ``1/L``.
    Convergence rate is O(1/k²) (Beck & Teboulle 2009).

    Parameters
    ----------
    operator : ForwardOperatorProtocol
        The grism forward operator H.
    noise_model : NoiseModel, optional
        Noise model for precision weights. Uses uniform weights if
        ``None``.
    lam : float
        Group-L1 regularisation strength λ. Default 1e-2.
    max_iter : int
        Maximum number of FISTA iterations. Default 200.
    lipschitz_n_iter : int
        Power iteration steps for step-size estimation. Default 30.
    tol : float
        Relative convergence tolerance.  Stops early when
        ``‖a_new − a‖ / (‖a‖ + 1e-10) < tol``.  Set to ``0.0``
        (default) to always run ``max_iter`` iterations.
    restart : bool
        Enable gradient-based adaptive restart (O'Donoghue & Candès
        2015).  When the inner product ``⟨∇f(y_k),  x_k − x_{k-1}⟩``
        is positive — indicating momentum overshoot — the momentum
        coefficient is reset to zero and iteration resumes from the
        current point.  Default ``True``.
    callback : callable, optional
        If provided, called at the end of every iteration as
        ``callback(iter, x, weighted_residual)`` where *iter* is
        1-indexed, *x* is the current coefficient array (do not
        mutate), and *weighted_residual* is ``‖W(Hx − f)‖``.
        Adds one extra ``apply()`` call per iteration when set.
        Default ``None``.

    Notes
    -----
    **Why gradient restart, not monotone FISTA or backtracking?**
    Gradient restart costs one dot product per iteration (O(K*M)).
    Monotone FISTA (MFISTA) requires an extra ``apply()`` call every
    time the objective increases; backtracking requires 1–3 extra
    calls per step.  For NIRISS WFSS data, ``H^T W² H`` is
    ill-conditioned (bright and faint sources coexist; overlapping
    traces; precision weights spanning orders of magnitude).  In this
    regime vanilla FISTA momentum overshoots the minimiser.  Restart
    directly addresses this failure mode at negligible cost.

    **Why fixed step 1/L, not backtracking?**
    ``power_iteration`` with 30 steps gives an accurate Lipschitz
    estimate for JAX operators.  Backtracking is only warranted when
    the estimate is unreliable; increase ``lipschitz_n_iter`` for
    atypical operators if needed.

    **Why are FISTA data residuals higher than LSQR?**
    LSQR minimises ``‖W(Hx − f)‖²`` without regularisation.  FISTA
    minimises the same term *plus* ``λ Σ_k ‖a_k‖₂``.  A non-zero λ
    moves the solution away from the least-squares minimum — that is
    the point (source deblending via group sparsity).  The relevant
    quality metric is spectrum RMSE, not data residual.
    """

    def __init__(
        self,
        operator: ForwardOperatorProtocol,
        noise_model=None,
        lam: float = 1e-2,
        max_iter: int = 200,
        lipschitz_n_iter: int = 30,
        tol: float = 0.0,
        restart: bool = True,
        callback=None,
    ) -> None:
        self._operator = operator
        self._noise_model = noise_model
        self._lam = lam
        self._max_iter = max_iter
        self._lipschitz_n_iter = lipschitz_n_iter
        self._tol = tol
        self._restart = restart
        self._callback = callback
        self._step: float | None = None  # computed lazily on first solve
```

- [ ] **Step 2: Verify file still parses**

```bash
cd /path/to/spectrex && uv run python -c "from spectrex.jax_solver import JAXProximalSolver; print('OK')"
```
Expected: `OK`

---

## Task 2: Update `JAXProximalSolver.solve()` with restart, tol, callback

**Files:**
- Modify: `src/spectrex/jax_solver.py:151-218`

- [ ] **Step 1: Replace the full `solve()` method**

Replace everything from `def solve(` through the closing `return a.astype(np.float32)` with:

```python
    def solve(
        self,
        dispersed: np.ndarray,
        precision_weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """Run FISTA to recover source coefficients.

        Parameters
        ----------
        dispersed : np.ndarray
            Dispersed detector image, shape ``image_shape`` or flat
            ``(n_pix,)``.
        precision_weights : np.ndarray, optional
            Per-pixel weights ``w = 1/σ``, shape ``(n_pix,)``. If
            ``None``, uses ``noise_model.precision_weights(dispersed)``
            when a noise model was provided; otherwise uniform weights.

        Returns
        -------
        np.ndarray
            Coefficient vector ``a``, shape ``(n_coefficients,)``,
            dtype ``float32``.
        """
        f = np.asarray(dispersed, dtype=np.float64).ravel()
        n_pix = f.size
        n_coef = self._operator.n_coefficients

        # Infer K and M for group-prox (JAXOperator exposes these;
        # fall back to treating all coefficients as one group).
        K = getattr(self._operator, "n_active", n_coef)
        M = getattr(self._operator, "n_components", 1)

        if precision_weights is not None:
            w = np.asarray(precision_weights, dtype=np.float64)
        elif self._noise_model is not None:
            w = self._noise_model.precision_weights(f)
        else:
            w = np.ones(n_pix, dtype=np.float64)

        step = self._get_step(w)

        # FISTA initialisation
        a = np.zeros(n_coef, dtype=np.float64)
        y = a.copy()
        t = 1.0

        for i in range(self._max_iter):
            # Gradient of (1/2)||W(Hy − f)||²: H^T W^2 (Hy − f)
            residual_w = w * (self._operator.apply(y).astype(np.float64) - f)
            grad = self._operator.apply_adjoint(w * residual_w).astype(np.float64)

            # Proximal gradient step
            v = y - step * grad
            a_new = group_soft_threshold(
                v.astype(np.float32), threshold=step * self._lam, K=K, M=M
            ).astype(np.float64)

            # Gradient restart (O'Donoghue & Candès 2015).
            # At this point `a` is x_{k-1} (not yet updated).
            if self._restart and float(np.dot(grad, a_new - a)) > 0.0:
                t = 1.0
                y = a_new  # discard momentum; restart from current point
            else:
                # Standard FISTA momentum update
                t_new = (1.0 + np.sqrt(1.0 + 4.0 * t ** 2)) / 2.0
                y = a_new + ((t - 1.0) / t_new) * (a_new - a)
                t = t_new

            # Optional per-iteration callback — one extra apply() when set
            if self._callback is not None:
                wr = float(np.linalg.norm(
                    w * (self._operator.apply(a_new).astype(np.float64) - f)
                ))
                self._callback(i + 1, a_new, wr)

            # Relative convergence check
            if self._tol > 0.0:
                delta = float(np.linalg.norm(a_new - a))
                base = float(np.linalg.norm(a)) + 1e-10
                if delta / base < self._tol:
                    a = a_new
                    break

            a = a_new

        logger.debug(
            "FISTA done: %d iters, final ||W(Ha−f)||=%.3e",
            self._max_iter,
            float(np.linalg.norm(w * (self._operator.apply(a).astype(np.float64) - f))),
        )
        return a.astype(np.float32)
```

- [ ] **Step 2: Run existing tests — all must still pass**

```bash
cd /path/to/spectrex && uv run pytest unittests/test_jax_solver.py -v
```
Expected: all 10 existing tests pass (no new tests yet).

- [ ] **Step 3: Commit**

```bash
git add src/spectrex/jax_solver.py
git commit -m "feat: add adaptive restart, tol, and callback to JAXProximalSolver"
```

---

## Task 3: Add three new tests

**Files:**
- Modify: `unittests/test_jax_solver.py` (append after line 128)

- [ ] **Step 1: Write the three failing tests**

Append to `unittests/test_jax_solver.py`:

```python


# ── New tests for restart / tol / callback ──────────────────────────────────

def test_callback_called_correct_times(small_problem):
    """Callback is invoked exactly max_iter times; iter is 1-indexed; wr ≥ 0."""
    op, _, f = small_problem
    calls: list[tuple[int, float]] = []

    def cb(i: int, x: np.ndarray, wr: float) -> None:
        calls.append((i, wr))

    solver = JAXProximalSolver(op, lam=0.01, max_iter=10, tol=0.0, callback=cb)
    solver.solve(f)

    assert len(calls) == 10, f"Expected 10 calls, got {len(calls)}"
    assert calls[0][0] == 1, "First callback iter should be 1 (1-indexed)"
    assert calls[-1][0] == 10, "Last callback iter should be 10"
    assert all(wr >= 0.0 for _, wr in calls), "All weighted residuals must be ≥ 0"


def test_tol_early_stopping(small_problem):
    """tol > 0 causes solve() to stop before max_iter when converged."""
    op, _, f = small_problem
    calls: list[int] = []

    def cb(i: int, x: np.ndarray, wr: float) -> None:
        calls.append(i)

    # Well-conditioned problem with lam=0 converges quickly — well under 500 iters
    solver = JAXProximalSolver(op, lam=0.0, max_iter=500, tol=1e-4, callback=cb)
    solver.solve(f)

    assert len(calls) < 500, (
        f"Expected early stop before 500 iters, ran {len(calls)}"
    )


def test_restart_does_not_degrade_solution(small_problem):
    """restart=True produces equivalent solution quality to restart=False."""
    op, a_true, f = small_problem

    a_no  = JAXProximalSolver(op, lam=0.0, max_iter=100, restart=False).solve(f)
    a_yes = JAXProximalSolver(op, lam=0.0, max_iter=100, restart=True).solve(f)

    rmse_no  = float(np.sqrt(np.mean((a_no  - a_true) ** 2)))
    rmse_yes = float(np.sqrt(np.mean((a_yes - a_true) ** 2)))

    # restart must not make reconstruction worse (allow 10% slack for numerical noise)
    assert rmse_yes <= rmse_no * 1.1 + 1e-4, (
        f"restart=True RMSE {rmse_yes:.4f} worse than restart=False {rmse_no:.4f}"
    )
```

- [ ] **Step 2: Run new tests — all three must pass**

```bash
cd /path/to/spectrex && uv run pytest unittests/test_jax_solver.py -v -k "callback or tol or restart"
```
Expected: 3 new tests PASS.

- [ ] **Step 3: Run full test suite**

```bash
cd /path/to/spectrex && uv run pytest unittests/ -v -m "not slow"
```
Expected: all 73 tests pass.

- [ ] **Step 4: Commit**

```bash
git add unittests/test_jax_solver.py
git commit -m "test: add callback, tol, and restart tests for JAXProximalSolver"
```

---

## Task 4: Fix `comparison_computational.ipynb`

**Files:**
- Modify: `notebooks/comparison_computational.ipynb` (cells 19, 20, 21, 22)

The notebook is JSON. Edit it with a Python script — do not edit by hand.

- [ ] **Step 1: Run the update script**

Create and run this once-off script (do not commit it):

```python
#!/usr/bin/env python3
"""Patch comparison_computational.ipynb: fix FISTA residual metric."""
import json
from pathlib import Path

NB_PATH = Path("notebooks/comparison_computational.ipynb")
nb = json.loads(NB_PATH.read_text())
cells = nb["cells"]


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def md_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


# ── Cell 19: update markdown header ─────────────────────────────────────────
assert cells[19]["cell_type"] == "markdown"
cells[19]["source"] = (
    "### FISTA Convergence (per-iteration weighted residual)\n\n"
    "We use the `callback` hook so the solver tracks its own weighted\n"
    "residual `‖W(Hx − f)‖` — the **same metric as LSQR** above.\n"
    "No reimplementation of the inner loop."
)

# ── Cell 20: replace manual FISTA loop with callback-based version ───────────
assert cells[20]["cell_type"] == "code"
cells[20]["source"] = """\
# FISTA warm-up (JIT compile) — excludes compilation from timing
_ = JAXProximalSolver(
    jax_op_conv, noise_model=NOISE_MODEL, lam=0.05, max_iter=1
).solve(f_noisy_conv)

# Per-iteration data collected via callback
fista_times: list[float] = []
fista_residuals: list[float] = []

def _fista_cb(i: int, x, wr: float) -> None:
    fista_times.append(time.perf_counter() - _t0)
    fista_residuals.append(wr)   # ‖W(Hx − f)‖ — same metric as LSQR

N_FISTA_ITER = 200
LAM = 0.05

solver = JAXProximalSolver(
    jax_op_conv, noise_model=NOISE_MODEL,
    lam=LAM, max_iter=N_FISTA_ITER,
    restart=True, tol=0.0,   # tol=0 runs exactly N_FISTA_ITER iterations
    callback=_fista_cb,
)
_t0 = time.perf_counter()
solver.solve(f_noisy_conv)
print(f'FISTA: {len(fista_residuals)} iters, '
      f'final weighted residual={fista_residuals[-1]:.4f}')
"""
cells[20]["outputs"] = []

# ── Cell 21: update markdown — note on fair comparison ──────────────────────
assert cells[21]["cell_type"] == "markdown"
cells[21]["source"] = (
    "### Residual Norm vs Wall-Clock Time\n\n"
    "Both curves now show `‖W(Hx − f)‖` (precision-weighted residual).\n\n"
    "> **Note:** FISTA's residual floor is expected to be higher than LSQR's.\n"
    "> LSQR minimises `‖W(Hx − f)‖²` with no regularisation; FISTA minimises\n"
    "> the same term *plus* `λ Σ_k ‖a_k‖₂`.  Non-zero λ trades data fit for\n"
    "> source deblending.  The relevant quality metric is spectrum RMSE\n"
    "> (see `comparison_solver_accuracy.ipynb`), not data residual.\n"
    ">\n"
    "> Adaptive restart (`restart=True`) resets momentum when the inner\n"
    "> product `⟨∇f(y_k), x_k − x_{k-1}⟩ > 0` — a sign of overshoot on\n"
    "> ill-conditioned operators.  Short plateaus in the FISTA curve mark\n"
    "> restart events."
)

# ── Cell 22: update axis label ───────────────────────────────────────────────
# Notebook source is stored as a list of strings; join → replace → re-split.
assert cells[22]["cell_type"] == "code"
src22 = "".join(cells[22]["source"])
src22 = src22.replace(
    "ax.set_ylabel('Residual norm (log scale)')",
    "ax.set_ylabel('‖W(Hx − f)‖  (weighted residual norm, log scale)')",
)
cells[22]["source"] = src22
cells[22]["outputs"] = []

NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print("Patched OK")
```

Run it:
```bash
cd /path/to/spectrex && uv run python /tmp/patch_notebook.py
```
Expected: `Patched OK`

- [ ] **Step 2: Re-execute the notebook**

```bash
cd /path/to/spectrex && uv run jupyter nbconvert \
    --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=600 \
    notebooks/comparison_computational.ipynb
```
Expected: exits 0, notebook has outputs in all code cells.

- [ ] **Step 3: Spot-check the outputs**

```bash
uv run python -c "
import json
nb = json.load(open('notebooks/comparison_computational.ipynb'))
cells = nb['cells']
# Cell 20: check final weighted residual printed
out20 = ''.join(o.get('text','') for o in cells[20]['outputs'])
print('Cell 20 output:', out20.strip())
# Cell 22: check axis label updated
print('Cell 22 ylabel present:', 'weighted residual norm' in ''.join(cells[22]['source']))
"
```
Expected output contains `FISTA: 200 iters, final weighted residual=` and a number substantially lower than `1028` (old unweighted figure). The axis-label check prints `True`.

- [ ] **Step 4: Commit**

```bash
git add notebooks/comparison_computational.ipynb
git commit -m "fix: correct FISTA residual metric in convergence notebook; use callback; document algorithmic choices"
```

---

## Task 5: Final verification

- [ ] **Step 1: Full fast test suite**

```bash
cd /path/to/spectrex && uv run pytest unittests/ -v -m "not slow"
```
Expected: **73 tests pass, 0 fail.**

- [ ] **Step 2: Public API still importable**

```bash
uv run python -c "
from spectrex import JAXProximalSolver
import inspect
sig = inspect.signature(JAXProximalSolver.__init__)
params = list(sig.parameters)
assert 'tol' in params, 'tol missing'
assert 'restart' in params, 'restart missing'
assert 'callback' in params, 'callback missing'
print('API OK:', params)
"
```
Expected: prints `API OK: [...]` with all three new params.

- [ ] **Step 3: Verify notebook has no missing outputs**

```bash
uv run python -c "
import json
nb = json.load(open('notebooks/comparison_computational.ipynb'))
code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
missing = [i for i, c in enumerate(code_cells) if not c.get('outputs')]
print('Missing outputs in code cells:', missing)
assert not missing, f'Cells {missing} have no outputs'
print('All code cells have outputs ✓')
"
```
Expected: `All code cells have outputs ✓`

- [ ] **Step 4: Final commit (if anything was touched during verification)**

```bash
git status  # should be clean; if not, commit any fix
```
