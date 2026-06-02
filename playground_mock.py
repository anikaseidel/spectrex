from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix
from matplotlib.patches import Rectangle

import spectrex
from spectrex import (
    EigenspectraBasis,
    InstrumentConfig,
    NoiseModel,
    SciPySparseOperator,
    SpectralSolver,
)

# ── Paths ────────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
TESTDATA = HERE/"testdata"
OPERATOR_CACHE = Path("operator_cache.npz")

# ── Configuration ─────────────────────────────────────────────────────────────
COLD_START = False    # set True to force operator rebuild from scratch
IMAGE_SHAPE = (500, 20) # Main frame
DETECTOR_SHAPE = (900,40) # Extended frame
SOURCE_ORIGIN = (200,10) # (0,0) of Detector starts at (10,200)
SOURCE_DENSITY = 0.05  # fraction of pixels with injected sources
SEED = 50
N_COMPONENTS = 10     # must match eigenspectra CSV

rng = np.random.default_rng(SEED)

print(f"spectrex {spectrex.__version__}")

# ── Instrument configuration & eigenspectra basis ───────────────────────────────────────

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

# ── Forward operator ──────────────────────────────────────────────────────

import time

if OPERATOR_CACHE.exists() and not COLD_START:
    op = SciPySparseOperator.load(OPERATOR_CACHE)
    print(f"Operator loaded from {OPERATOR_CACHE}  "
          f"(shape {op._H.shape[0]} × {op._H.shape[1]})")
else:
    t0 = time.perf_counter()
    op = SciPySparseOperator.build_extended(config, basis, IMAGE_SHAPE,DETECTOR_SHAPE,SOURCE_ORIGIN)
    op.save(OPERATOR_CACHE)
    elapsed = time.perf_counter() - t0
    print(f"Operator built in {elapsed:.1f} s — cached to {OPERATOR_CACHE}")
    print(f"Shape: {op._H.shape[0]} × {op._H.shape[1]}")

# Helper
def _clip(arr, nsigma_lo=2, nsigma_hi=2):
    m, s = np.nanmean(arr), np.nanstd(arr)
    return m - nsigma_lo * s, m + nsigma_hi * s   

# With new M construction
def run_mock_scene_optimized_recovery(
    SOURCE_DENSITY,
    op,
    basis,
    IMAGE_SHAPE,
    DETECTOR_SHAPE,
    rng,
    solver_kwargs=None,
    PARITY=None,
    PLOTS=None,
    MAX_TRIES=50,
    SIGMA_MEAN=0.5,
    SIGMA_STD=0.25,
    RADIUS_FACTOR=2,
):
    """
    Create a random mock scene, recover it, and compute reconstruction metrics.

    Returns
    -------
    result : dict
        Contains:
            - density
            - mse
            - recovered
            - a_tilde
            - direct
            - dispersed
            - recovered_img
            - residual_img
            - all_true_vals
            - all_rec_vals
    """

    if solver_kwargs is None:
        solver_kwargs = dict(max_iter=500, tolerance=1e-8)

    H, W = DETECTOR_SHAPE
    K,L = IMAGE_SHAPE
    n_pix_src = K*L
    n_pix_det = H * W
    n = basis.n_components

    a_tilde = np.zeros(n_pix_det * n)

    num_active = int(SOURCE_DENSITY * n_pix_det)
    active_k = rng.choice(n_pix_det, size=num_active, replace=False)

    # -------------------------------------------------------------------------
    # Source creation
    # -------------------------------------------------------------------------

    sources = {}
    pixel_to_source = -np.ones(n_pix_det, dtype=int)

    source_id = 0

    for k in active_k:

        y0, x0 = divmod(k, W)

        for _ in range(MAX_TRIES):

            flux = rng.uniform(-1, 1, size=n)

            # require positive reconstructed spectrum
            if not np.all(basis.reconstruct(flux) >= 0):
                continue

            sigma = max(0.5, rng.normal(SIGMA_MEAN, SIGMA_STD))
            r = int(np.ceil(RADIUS_FACTOR * sigma))

            y_min = max(0, y0 - r)
            y_max = min(H, y0 + r + 1)

            x_min = max(0, x0 - r)
            x_max = min(W, x0 + r + 1)

            ys = np.arange(y_min, y_max)
            xs = np.arange(x_min, x_max)

            YY, XX = np.meshgrid(ys, xs, indexing="ij")

            gauss = np.exp(
                -((XX - x0) ** 2 + (YY - y0) ** 2) / (2 * sigma**2)
            )

            gauss /= gauss.max()

            pixels = []
            amplitudes = []

            for yy, xx, amp in zip(YY.ravel(), XX.ravel(), gauss.ravel()):

                kk = yy * W + xx

                pixels.append(kk)
                amplitudes.append(amp)

                pixel_to_source[kk] = source_id

                a_tilde[kk * n : (kk + 1) * n] += amp * flux

            sources[source_id] = {
                "center": (y0, x0),
                "flux": flux,
                "amplitudes": np.array(amplitudes),
                "sigma": sigma,
                "pixels": np.array(pixels, dtype=int),
            }

            source_id += 1
            break

    print(f"Sources placed: {len(sources)}")

    # -------------------------------------------------------------------------
    # Images
    # -------------------------------------------------------------------------

    direct = basis.broadband_image(a_tilde, DETECTOR_SHAPE)

    dispersed = op.apply(a_tilde).reshape(IMAGE_SHAPE)

    # -------------------------------------------------------------------------
    # Mixing matrix
    # -------------------------------------------------------------------------

    n_src = len(sources)
    rows = []
    cols = []
    data = []

    #M = np.zeros((n_pix_det * n, n_src * n))

    for source_id, s in sources.items():

        amplitudes = s["amplitudes"]
        pixels = s["pixels"]

        for amp, kk in zip(amplitudes, pixels):

            for c in range(n):

                rows.append(kk * n + c)
                cols.append(source_id * n + c)
                data.append(amp)

    M = coo_matrix(
        (data, (rows, cols)),
        shape=(n_pix_det * n, n_src * n)
    )

    # Often convert to CSR for efficient arithmetic
    M = M.tocsr()
    # -------------------------------------------------------------------------
    # Recovery
    # -------------------------------------------------------------------------

    support_mask = a_tilde != 0

    blocks = support_mask.reshape(-1, n)

    active_blocks = blocks.any(axis=1)

    num_active_blocks = active_blocks.sum()

    print(f"Dispersed image range: [{dispersed.min():.4f}, {dispersed.max():.4f}]")
    print(f"Active coefficients: {support_mask.sum()} / {len(support_mask)}")
    print(f"Pixel density: {(num_active_blocks / (H * W)) * 100:.4f}%")
    print(f"Source density: {(len(sources) / (H * W)) * 100:.4f}%")

    solver = SpectralSolver(op, **solver_kwargs)

    recovered = solver.solve(
        dispersed,
        support_mask=support_mask,
        M=M,
    )

    print(f"Recovered vector shape: {recovered.shape}")

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------

    
    if PARITY == True:
        # only sources in main frame are evaluated!
        active_indices = []

        for k in range(n_pix_det):

            if not np.any(a_tilde[k * n:(k + 1) * n] != 0):
                continue

            row = k // W
            col = k % W

            if (
                SOURCE_ORIGIN[1] <= col < SOURCE_ORIGIN[1] + IMAGE_SHAPE[1]
                and
                SOURCE_ORIGIN[0] <= row < SOURCE_ORIGIN[0] + IMAGE_SHAPE[0]
            ):
                active_indices.append(k)
                
        true_flux = np.concatenate(
            [basis.reconstruct(a_tilde[k * n : (k + 1) * n]) for k in active_indices]
        )
        rec_flux = np.concatenate(
            [basis.reconstruct(recovered[k * n : (k + 1) * n]) for k in active_indices]
        )


        fig, axes = plt.subplots(2, 1, figsize=(5, 8), sharex=True, tight_layout=True, height_ratios=(1, 0.6), gridspec_kw={'hspace': 0})
        ax = axes[0]
        outliers = rec_flux <= 1.

        ax.scatter(true_flux[~outliers], rec_flux[~outliers], s=2, alpha=0.05, linewidths=0, color="C0", rasterized=True)
        ax.scatter(true_flux[outliers], rec_flux[outliers], s=2, alpha=0.05, linewidths=0, color="C1", rasterized=True)
        minv = max(min(true_flux.min(), rec_flux.min()), -10_000)
        maxv = max(true_flux.max(), rec_flux.max())
        ax.plot([minv, maxv], [minv, maxv], "r--", lw=1, label="1:1")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(minv, maxv)
        ax.set_ylim(minv, maxv)
        ax.set_ylabel("Recovered flux  f(λ)")
        ax.set_title("Parity plot — noiseless")

        ax = axes[1]
        frac_residuals = (true_flux - rec_flux) / true_flux
        ax.scatter(true_flux[~outliers], frac_residuals[~outliers], s=2, alpha=0.05, linewidths=0, color="C0", rasterized=True)
        ax.scatter(true_flux[outliers], frac_residuals[outliers], s=2, alpha=0.05, linewidths=0, color="C1", rasterized=True)
        ax.set_ylim(-2, 2)
        ax.set_xlabel("True flux  f(λ)")
        rmse_noiseless_good = np.sqrt(np.mean(frac_residuals[~outliers]) ** 2)
        ax.text(0.05, 0.92, f"frac. mean error = {rmse_noiseless_good:.4f}", color='C0',
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        fig.tight_layout()


        plt.show()
    if PLOTS == True:
        active_indices = [k for k in range(n_pix_det) if np.any(a_tilde[k * n : (k + 1) * n] != 0)]

        true_flux = np.concatenate(
            [basis.reconstruct(a_tilde[k * n : (k + 1) * n]) for k in active_indices]
        )
        rec_flux = np.concatenate(
            [basis.reconstruct(recovered[k * n : (k + 1) * n]) for k in active_indices]
        )

        rmse_noiseless = np.sqrt(np.mean((true_flux - rec_flux) ** 2))
        print(f"Noiseless RMSE (flux): {rmse_noiseless:.6f}")


        recovered_img = basis.broadband_image(recovered, DETECTOR_SHAPE)
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
            #ax.set_aspect('equal', adjustable='box')  # scaled axes
            if title == "Direct image" or title == "Recovered" or title == "|Residual|":
                # Darken everything that is not detector region
                alpha = np.full(im.get_array().shape, 0.6)
                alpha[SOURCE_ORIGIN[0]:(IMAGE_SHAPE[0]+SOURCE_ORIGIN[0]), SOURCE_ORIGIN[1]:(IMAGE_SHAPE[1]+SOURCE_ORIGIN[1])] = 0.0

                ax.imshow(
                    np.zeros_like(im.get_array()),
                    cmap="gray",
                    alpha=alpha,
                    aspect="auto",
                    origin=im.origin if hasattr(im, "origin") else None,
                )
                # Red rectangle around the ROI
                ax.add_patch(
                    Rectangle(
                        (SOURCE_ORIGIN[1], SOURCE_ORIGIN[0]),      # (x_min, y_min)
                        IMAGE_SHAPE[1],             # width  = 30 - 10
                        IMAGE_SHAPE[0],            # height = 700 - 200
                        fill=False,
                        edgecolor="red",
                        linewidth=1,
                    )
                )
            
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            

        fig.suptitle("Noiseless recovery — 500 × 20 stamp, GR150R/F150W", y=1.01)
        fig.tight_layout()
        plt.show()
    
    recovered_img = basis.broadband_image(recovered, DETECTOR_SHAPE)
    residual_img  = np.abs(direct - recovered_img)
    
    all_true_vals = []
    all_rec_vals = []

    count = 0
    norm = 0

    x_pixel = W

    for i in range(int(len(a_tilde) / n)):

        if np.any(a_tilde[i * n : (i + 1) * n] != 0):

            row = i // x_pixel
            col = i % x_pixel

            if col >= SOURCE_ORIGIN[1] and col <(IMAGE_SHAPE[1]+SOURCE_ORIGIN[1]):

                if row >= SOURCE_ORIGIN[0] and row <(IMAGE_SHAPE[0]+SOURCE_ORIGIN[0]):

                    count += 1

                    spectrum = basis.reconstruct(recovered[i * n : (i + 1) * n])
                    spectrum_og = basis.reconstruct(a_tilde[i * n : (i + 1) * n])

                    norm += (
                        np.linalg.norm(spectrum - spectrum_og)
                        / (np.linalg.norm(spectrum_og) + 1e-8)
                    )

                    all_true_vals.append(spectrum_og)
                    all_rec_vals.append(spectrum)

    pixel_dens = np.sum(direct[SOURCE_ORIGIN[0]:(IMAGE_SHAPE[0]+SOURCE_ORIGIN[0]), SOURCE_ORIGIN[1]:(IMAGE_SHAPE[1]+SOURCE_ORIGIN[1])] != 0)/ (((IMAGE_SHAPE[1]+SOURCE_ORIGIN[1]) - SOURCE_ORIGIN[1]) * ((IMAGE_SHAPE[0]+SOURCE_ORIGIN[0])-SOURCE_ORIGIN[0]))
    average = norm / count if count > 0 else 0

    print("Pixel density of recoverable sources:", pixel_dens)
    print("MSE =", average)

    return {
        "pixel density": pixel_dens,
        "mse": average,
        "recovered": recovered,
        "a_tilde": a_tilde,
        "direct": direct,
        "dispersed": dispersed,
        "recovered_img": recovered_img,
        "residual_img": residual_img,
        "all_true_vals": all_true_vals,
        "all_rec_vals": all_rec_vals,
    }
    
# old version with matrix pruning
def run_mock_scene_recovery(
    SOURCE_DENSITY,
    op,
    basis,
    IMAGE_SHAPE,
    rng,
    solver_kwargs=None,
    PARITY= None,
    PLOTS=None,
    MAX_TRIES=50,
    SIGMA_MEAN=0.5,
    SIGMA_STD=0.25,
    RADIUS_FACTOR=2,
):
    """
    Create a random mock scene, recover it, and compute reconstruction metrics.
    Without mixing operator M.

    Returns
    -------
    result : dict
        Contains:
            - density
            - mse
            - recovered
            - a_tilde
            - direct
            - dispersed
            - recovered_img
            - residual_img
            - all_true_vals
            - all_rec_vals
    """

    if solver_kwargs is None:
        solver_kwargs = dict(max_iter=500, tolerance=1e-8)

    H, W = IMAGE_SHAPE
    n_pix = H * W
    n = basis.n_components

    a_tilde = np.zeros(n_pix * n)

    num_active = int(SOURCE_DENSITY * n_pix)
    active_k = rng.choice(n_pix, size=num_active, replace=False)

    # -------------------------------------------------------------------------
    # Source creation
    # -------------------------------------------------------------------------

    sources = {}
    pixel_to_source = -np.ones(n_pix, dtype=int)

    source_id = 0

    for k in active_k:

        y0, x0 = divmod(k, W)

        for _ in range(MAX_TRIES):

            flux = rng.uniform(-1, 1, size=n)

            # require positive reconstructed spectrum
            if not np.all(basis.reconstruct(flux) >= 0):
                continue

            sigma = max(0.5, rng.normal(SIGMA_MEAN, SIGMA_STD))
            r = int(np.ceil(RADIUS_FACTOR * sigma))

            y_min = max(0, y0 - r)
            y_max = min(H, y0 + r + 1)

            x_min = max(0, x0 - r)
            x_max = min(W, x0 + r + 1)

            ys = np.arange(y_min, y_max)
            xs = np.arange(x_min, x_max)

            YY, XX = np.meshgrid(ys, xs, indexing="ij")

            gauss = np.exp(
                -((XX - x0) ** 2 + (YY - y0) ** 2) / (2 * sigma**2)
            )

            gauss /= gauss.max()

            pixels = []
            amplitudes = []

            for yy, xx, amp in zip(YY.ravel(), XX.ravel(), gauss.ravel()):

                kk = yy * W + xx

                pixels.append(kk)
                amplitudes.append(amp)

                pixel_to_source[kk] = source_id

                a_tilde[kk * n : (kk + 1) * n] += amp * flux

            sources[source_id] = {
                "center": (y0, x0),
                "flux": flux,
                "amplitudes": np.array(amplitudes),
                "sigma": sigma,
                "pixels": np.array(pixels, dtype=int),
            }

            source_id += 1
            break

    print(f"Sources placed: {len(sources)}")

    # -------------------------------------------------------------------------
    # Images
    # -------------------------------------------------------------------------

    direct = basis.broadband_image(a_tilde, IMAGE_SHAPE)

    dispersed = op.apply(a_tilde).reshape(IMAGE_SHAPE)


    # -------------------------------------------------------------------------
    # Recovery
    # -------------------------------------------------------------------------

    support_mask = a_tilde != 0

    blocks = support_mask.reshape(-1, n)

    active_blocks = blocks.any(axis=1)

    num_active_blocks = active_blocks.sum()

    print(f"Dispersed image range: [{dispersed.min():.4f}, {dispersed.max():.4f}]")
    print(f"Active coefficients: {support_mask.sum()} / {len(support_mask)}")
    print(f"Pixel density: {(num_active_blocks / (H * W)) * 100:.4f}%")
    print(f"Source density: {(len(sources) / (H * W)) * 100:.4f}%")

    solver = SpectralSolver(op, **solver_kwargs)

    recovered = solver.solve(
        dispersed,
        support_mask=support_mask,
        M=None,
    )

    print(f"Recovered vector shape: {recovered.shape}")

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------

    
    if PARITY == True:
        active_indices = [k for k in range(n_pix) if np.any(a_tilde[k * n : (k + 1) * n] != 0)]

        true_flux = np.concatenate(
            [basis.reconstruct(a_tilde[k * n : (k + 1) * n]) for k in active_indices]
        )
        rec_flux = np.concatenate(
            [basis.reconstruct(recovered[k * n : (k + 1) * n]) for k in active_indices]
        )


        fig, axes = plt.subplots(2, 1, figsize=(5, 8), sharex=True, tight_layout=True, height_ratios=(1, 0.6), gridspec_kw={'hspace': 0})
        ax = axes[0]
        outliers = rec_flux <= 1.

        ax.scatter(true_flux[~outliers], rec_flux[~outliers], s=2, alpha=0.05, linewidths=0, color="C0", rasterized=True)
        ax.scatter(true_flux[outliers], rec_flux[outliers], s=2, alpha=0.05, linewidths=0, color="C1", rasterized=True)
        minv = max(min(true_flux.min(), rec_flux.min()), -10_000)
        maxv = max(true_flux.max(), rec_flux.max())
        ax.plot([minv, maxv], [minv, maxv], "r--", lw=1, label="1:1")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(minv, maxv)
        ax.set_ylim(minv, maxv)
        ax.set_ylabel("Recovered flux  f(λ)")
        ax.set_title("Parity plot — noiseless")

        ax = axes[1]
        frac_residuals = (true_flux - rec_flux) / true_flux
        ax.scatter(true_flux[~outliers], frac_residuals[~outliers], s=2, alpha=0.05, linewidths=0, color="C0", rasterized=True)
        ax.scatter(true_flux[outliers], frac_residuals[outliers], s=2, alpha=0.05, linewidths=0, color="C1", rasterized=True)
        ax.set_ylim(-2, 2)
        ax.set_xlabel("True flux  f(λ)")
        rmse_noiseless_good = np.sqrt(np.mean(frac_residuals[~outliers]) ** 2)
        ax.text(0.05, 0.92, f"frac. mean error = {rmse_noiseless_good:.4f}", color='C0',
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        fig.tight_layout()


        plt.show()
    if PLOTS == True:
        active_indices = [k for k in range(n_pix) if np.any(a_tilde[k * n : (k + 1) * n] != 0)]

        true_flux = np.concatenate(
            [basis.reconstruct(a_tilde[k * n : (k + 1) * n]) for k in active_indices]
        )
        rec_flux = np.concatenate(
            [basis.reconstruct(recovered[k * n : (k + 1) * n]) for k in active_indices]
        )

        rmse_noiseless = np.sqrt(np.mean((true_flux - rec_flux) ** 2))
        print(f"Noiseless RMSE (flux): {rmse_noiseless:.6f}")


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
            #ax.set_aspect('equal', adjustable='box')  # scaled axes
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
       
        fig.suptitle("Noiseless recovery — 500 × 20 stamp, GR150R/F150W", y=1.01)
        fig.tight_layout()
        plt.show()

    recovered_img = basis.broadband_image(recovered, IMAGE_SHAPE)
    residual_img  = np.abs(direct - recovered_img)

    all_true_vals = []
    all_rec_vals = []

    count = 0
    norm = 0

    x_pixel = W

    for i in range(int(len(a_tilde) / n)):

        if np.any(a_tilde[i * n : (i + 1) * n] != 0):

            row = i // x_pixel
            col = i % x_pixel

            if col > 3:

                if row > 233 and row <400:

                    count += 1

                    spectrum = basis.reconstruct(recovered[i * n : (i + 1) * n])
                    spectrum_og = basis.reconstruct(a_tilde[i * n : (i + 1) * n])

                    norm += (
                        np.linalg.norm(spectrum - spectrum_og)
                        / (np.linalg.norm(spectrum_og) + 1e-8)
                    )

                    all_true_vals.append(spectrum_og)
                    all_rec_vals.append(spectrum)

    pixel_dens = np.sum(direct[233:400, 3:] != 0)/ ((W - 3) * (400-233))
    average = norm / count if count > 0 else 0

    print("Pixel density of recoverable sources:", pixel_dens)
    print("MSE =", average)

    return {
        "pixel density": pixel_dens,
        "mse": average,
        "recovered": recovered,
        "a_tilde": a_tilde,
        "direct": direct,
        "dispersed": dispersed,
        "recovered_img": recovered_img,
        "residual_img": residual_img,
        "all_true_vals": all_true_vals,
        "all_rec_vals": all_rec_vals,
    }
 
 
 
# Loop over several densities   
def run_densities(
    MAX_DENSITY,
    STEPS,
    op,
    basis,
    IMAGE_SHAPE,
    rng,
    OPTIMIZED = True,
    ax1 = None,
):
    pixel_dens_vals = []
    mse_vals = []

    all_true_vals_global = []
    all_rec_vals_global = []

    factor = MAX_DENSITY/STEPS
    for i in range(1,STEPS+1):
        sd = factor*i
        print("=" * 80)
        print(f"Running source density = {sd:.4f}")

        if OPTIMIZED == True:
            result = run_mock_scene_optimized_recovery(
                SOURCE_DENSITY=sd,
                op=op,
                basis=basis,
                IMAGE_SHAPE=IMAGE_SHAPE,
                rng=rng,
            )
        else:
            result = run_mock_scene_recovery(
            SOURCE_DENSITY=sd,
            op=op,
            basis=basis,
            IMAGE_SHAPE=IMAGE_SHAPE,
            rng=rng,
        )

        #density_vals.append(result["density"])
        pixel_dens_vals.append(result["pixel density"])
        mse_vals.append(result["mse"])

        all_true_vals_global.extend(result["all_true_vals"])
        all_rec_vals_global.extend(result["all_rec_vals"])
        
    pairs = sorted(zip(pixel_dens_vals, mse_vals))

    pixel_density_sorted, mse_sorted = zip(*pairs)

    if ax1 is None:
        fig, ax1 = plt.subplots(figsize=(7, 5))
    else:
        fig = ax1.figure
    # ------------------------------------------------------------------
    # main curve: pixel density vs error
    # ------------------------------------------------------------------

    ax1.plot(
        pixel_density_sorted,
        mse_sorted,
        marker="o",
        linewidth=2,
    )

    ax1.set_xlabel("Pixel density", fontsize=14)
    ax1.set_ylabel("Mean Square Error", fontsize=14)

    ax1.tick_params(axis="both", which="major", labelsize=13)

    plt.tight_layout()

    return fig, ax1
################parity plots
SEED = 50
rng = np.random.default_rng(SEED)

run_mock_scene_optimized_recovery(
                SOURCE_DENSITY=0.01,
                op=op,
                basis=basis,
                IMAGE_SHAPE=IMAGE_SHAPE,
                DETECTOR_SHAPE=DETECTOR_SHAPE,
                rng=rng,
                PARITY=True,
                PLOTS=True,
            )

# SEED = 50
# rng = np.random.default_rng(SEED)

# run_mock_scene_recovery(
#                 SOURCE_DENSITY=0.1,
#                 op=op,
#                 basis=basis,
#                 IMAGE_SHAPE=IMAGE_SHAPE,
#                 rng=rng,
#                 PARITY=True,
#                 PLOTS=True,
#             )

############# density plot example
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# run_densities(
#     MAX_DENSITY=0.05,
#     STEPS=3,
#     op=op,
#     basis=basis,
#     IMAGE_SHAPE=IMAGE_SHAPE,
#     rng=rng,
#     OPTIMIZED= True,
#     ax1= axes[0],
# )


# run_densities(
#     MAX_DENSITY=0.05,
#     STEPS=3,
#     op=op,
#     basis=basis,
#     IMAGE_SHAPE=IMAGE_SHAPE,
#     rng=rng,
#     OPTIMIZED= False,
#     ax1 = axes[1],
# )
# axes[0].set_title("Optimized with Mixing Operator")
# axes[1].set_title("Non-optimized")

# plt.tight_layout()
# plt.show()