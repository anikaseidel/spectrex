"""
Real-image extension of the spectrex mock-scene pipeline.
==========================================================
Replaces the synthetic scene with:
  - A real direct FITS image  → source detection + Gaussian PSF fitting
  - A real dispersed FITS image → used as the observed grism frame

The coefficient vector  a_tilde  is built from the fitted Gaussian profiles
and eigenspectra projections, then passed to SpectralSolver exactly as in the
mock pipeline.

Usage
-----
    python spectrex_real_images.py direct.fits dispersed.fits \
        [--detection-sigma 5] [--fit-box 15] [--parity] [--plots]
"""

from pathlib import Path
import argparse
import time

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.modeling import models, fitting
from photutils.detection import DAOStarFinder
from astropy.convolution import interpolate_replace_nans, Gaussian2DKernel

import spectrex
from spectrex import (
    EigenspectraBasis,
    InstrumentConfig,
    NoiseModel,
    SciPySparseOperator,
    SpectralSolver,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE       = Path(__file__).resolve().parent
TESTDATA   = HERE / "testdata"
OPERATOR_CACHE = Path("operator_cache.npz")

# ── Instrument / basis setup (identical to mock pipeline) ────────────────────
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

N_COMPONENTS = basis.n_components


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clip(arr, nsigma_lo=2, nsigma_hi=2):
    m, s = np.nanmean(arr), np.nanstd(arr)
    return m - nsigma_lo * s, m + nsigma_hi * s


def load_fits_image(path: str) -> np.ndarray:
    """Return the first 2-D HDU as float32."""
    with fits.open(path) as hdul:
        for hdu in hdul:
            if hdu.data is not None and hdu.data.ndim == 2:
                return hdu.data.astype(np.float32)
    raise ValueError(f"No 2-D image extension found in {path}")


def subtract_background(image: np.ndarray, sigma: float = 3.0):
    """Sigma-clipped median background subtraction."""
    _, median, _ = sigma_clipped_stats(image, sigma=sigma)
    return image - median, median


# ── Contamination geometry ────────────────────────────────────────────────────

def build_contamination_mask(disp_shape, direct_shape,
                              dispersion_px_per_nm, lambda_min_nm,
                              lambda_max_nm, lambda_ref_nm,
                              tilt_deg=0.0, n_lambda=200):
    """
    True where a dispersed pixel cannot be back-projected to ANY source
    inside the direct frame (i.e. purely from an out-of-field contaminant).
    """
    ny_d, nx_d   = disp_shape
    ny_dir, nx_dir = direct_shape

    lambdas  = np.linspace(lambda_min_nm, lambda_max_nm, n_lambda)
    shifts_x = (lambdas - lambda_ref_nm) * dispersion_px_per_nm
    shifts_y = shifts_x * np.tan(np.deg2rad(tilt_deg))

    rows, cols = np.mgrid[0:ny_d, 0:nx_d]
    src_x = cols[None] - shifts_x[:, None, None]
    src_y = rows[None] - shifts_y[:, None, None]

    any_in = np.any(
        (src_x >= 0) & (src_x < nx_dir) &
        (src_y >= 0) & (src_y < ny_dir),
        axis=0,
    )
    return ~any_in   # True = contaminated


def filter_out_of_field_sources(detections, disp_shape, direct_shape,
                                  dispersion_px_per_nm, lambda_min_nm,
                                  lambda_max_nm, lambda_ref_nm, tilt_deg=0.0):
    """
    Remove detections whose zeroth-order lies outside the direct frame,
    or whose entire trace falls outside the dispersed frame.
    """
    ny_dir, nx_dir = direct_shape
    ny_disp, nx_disp = disp_shape

    lambdas  = np.linspace(lambda_min_nm, lambda_max_nm, 200)
    shifts_x = (lambdas - lambda_ref_nm) * dispersion_px_per_nm
    shifts_y = shifts_x * np.tan(np.deg2rad(tilt_deg))

    kept = []
    for (x0, y0, peak) in detections:
        if not (0 <= x0 < nx_dir and 0 <= y0 < ny_dir):
            continue
        tx = x0 + shifts_x
        ty = y0 + shifts_y
        if np.any((tx >= 0) & (tx < nx_disp) & (ty >= 0) & (ty < ny_disp)):
            kept.append((x0, y0, peak))
    return kept


# ── Source detection & PSF fitting ───────────────────────────────────────────

def detect_sources(image_bg, threshold_sigma=5.0, fwhm=3.0):
    _, _, std = sigma_clipped_stats(image_bg, sigma=3.0)
    dao = DAOStarFinder(fwhm=fwhm, threshold=threshold_sigma * std,
                        exclude_border=True)
    tbl = dao(image_bg)
    if tbl is None or len(tbl) == 0:
        return []
    return [(float(r['x_centroid']), float(r['y_centroid']), float(r['peak']))
            for r in tbl]


def fit_gaussian_psf(image_bg, detections, fit_box=15):
    """Fit a 2-D Gaussian to each detection; return list of param dicts."""
    ny, nx = image_bg.shape
    fitter = fitting.LevMarLSQFitter()
    half   = fit_box // 2
    sources = []

    for sid, (x0, y0, peak) in enumerate(detections):
        ix, iy = int(round(x0)), int(round(y0))
        x1, x2 = max(0, ix - half), min(nx, ix + half + 1)
        y1, y2 = max(0, iy - half), min(ny, iy + half + 1)
        stamp   = image_bg[y1:y2, x1:x2]
        yy, xx  = np.mgrid[y1:y2, x1:x2].astype(float)

        init = models.Gaussian2D(
            amplitude=peak, x_mean=x0, y_mean=y0,
            x_stddev=2.0, y_stddev=2.0, theta=0.0,
        )
        init.x_stddev.bounds = (0.5, half)
        init.y_stddev.bounds = (0.5, half)

        try:
            fit = fitter(init, xx, yy, stamp)
            sources.append(dict(
                id=sid,
                x=float(fit.x_mean.value), y=float(fit.y_mean.value),
                amplitude=float(fit.amplitude.value),
                sigma_x=float(fit.x_stddev.value),
                sigma_y=float(fit.y_stddev.value),
                theta=float(fit.theta.value),
            ))
        except Exception:
            sources.append(dict(
                id=sid, x=x0, y=y0, amplitude=peak,
                sigma_x=2.0, sigma_y=2.0, theta=0.0,
            ))

    return sources


# ── Build a_tilde from real detections ───────────────────────────────────────

def _gaussian_stamp(src, image_shape, radius_factor=2):
    """
    Return pixel indices and Gaussian amplitudes for one source.
    Mirrors exactly how the mock pipeline builds its Gaussian blob.
    """
    H, W   = image_shape
    x0, y0 = src['x'], src['y']
    sigma  = 0.5 * (src['sigma_x'] + src['sigma_y'])   # isotropic approximation
    r      = int(np.ceil(radius_factor * sigma))

    y_min, y_max = max(0, int(y0) - r), min(H, int(y0) + r + 1)
    x_min, x_max = max(0, int(x0) - r), min(W, int(x0) + r + 1)

    YY, XX = np.mgrid[y_min:y_max, x_min:x_max]
    gauss  = np.exp(-((XX - x0) ** 2 + (YY - y0) ** 2) / (2 * sigma ** 2))
    gauss /= gauss.max()

    pixels     = (YY.ravel() * W + XX.ravel()).astype(int)
    amplitudes = gauss.ravel()
    return pixels, amplitudes


def build_a_tilde_from_sources(sources, direct_bg, image_shape,
                                basis, radius_factor=2):
    H, W   = image_shape
    n      = basis.n_components
    n_pix  = H * W
    a_tilde = np.zeros(n_pix * n, dtype=np.float64)

    # Inspect what basis actually exposes
    print(f"    basis attributes: {[a for a in dir(basis) if not a.startswith('_')]}")

    for src in sources:
        pixels, amplitudes = _gaussian_stamp(src, image_shape, radius_factor)

        # Simple initialisation: unit coefficients scaled by source amplitude.
        # First component gets the amplitude, rest are zero.
        # This avoids needing to invert the basis entirely.
        coeff = np.zeros(n)
        coeff[0] = src['amplitude']

        for kk, amp in zip(pixels, amplitudes):
            if 0 <= kk < n_pix:
                a_tilde[kk * n : (kk + 1) * n] += amp * coeff

    return a_tilde

# ── Mixing matrix M (same structure as mock pipeline) ────────────────────────

def build_mixing_matrix(sources, image_shape, basis, radius_factor=2):
    H, W   = image_shape
    n      = basis.n_components
    n_pix  = H * W
    n_src  = len(sources)

    M = np.zeros((n_pix * n, n_src * n), dtype=np.float32)

    for j, src in enumerate(sources):
        pixels, amplitudes = _gaussian_stamp(src, image_shape, radius_factor)
        for kk, amp in zip(pixels, amplitudes):
            if 0 <= kk < n_pix:
                for c in range(n):
                    M[kk * n + c, j * n + c] = amp

    return M


# ── Main real-image pipeline ──────────────────────────────────────────────────

def run_real_image_pipeline(
    direct_path: str,
    dispersed_path: str,
    *,
    # grism geometry (tune to your instrument / filter)
    dispersion_px_per_nm: float = 4.65,
    lambda_min_nm: float        = 800.0,
    lambda_max_nm: float        = 1150.0,
    lambda_ref_nm: float        = 975.0,
    tilt_deg: float             = 0.0,
    # detection / fitting
    detection_sigma: float = 5.0,
    fit_box: int           = 15,
    radius_factor: float   = 2.0,
    # solver
    max_iter: int   = 500,
    tolerance: float = 1e-8,
    use_mixing: bool = True,
    # plots
    parity: bool = False,
    plots:  bool = False,
    cold_start: bool = False,
):
    print("=" * 65)
    print(f"spectrex {spectrex.__version__}  —  real-image pipeline")
    print("=" * 65)

    # ── 1. Load & background-subtract ────────────────────────────────────────
    print("\n[1] Loading FITS images …")
    direct_raw    = load_fits_image(direct_path)
    dispersed_raw = load_fits_image(dispersed_path)
    # ── Clip to a smaller region for testing ─────────────────────────
    direct_raw    = direct_raw   [0:400, 0:50]
    dispersed_raw = dispersed_raw[0:400, 0:50]
    # ─── NaNs ──────────────────────────────────────────────────────────────

    kernel = Gaussian2DKernel(x_stddev=1)
    direct_raw    = interpolate_replace_nans(direct_raw,    kernel)
    dispersed_raw = interpolate_replace_nans(dispersed_raw, kernel)
    IMAGE_SHAPE   = direct_raw.shape
    print(f"    Direct    : {direct_raw.shape}")
    print(f"    Dispersed : {dispersed_raw.shape}")

    direct_bg,    _ = subtract_background(direct_raw)
    dispersed_bg, _ = subtract_background(dispersed_raw)

    # ── 2. Operator (cached, same as mock) ────────────────────────────────────
    print("\n[2] Forward operator …")
    if OPERATOR_CACHE.exists() and not cold_start:
        op = SciPySparseOperator.load(OPERATOR_CACHE)
        print(f"    Loaded from cache  (shape {op._H.shape})")
    else:
        t0 = time.perf_counter()
        op = SciPySparseOperator.build(config, basis, IMAGE_SHAPE)
        op.save(OPERATOR_CACHE)
        print(f"    Built in {time.perf_counter()-t0:.1f} s")

    # ── 3. Source detection in direct image ───────────────────────────────────
    print(f"\n[3] Detecting sources (threshold = {detection_sigma}σ) …")
    raw_detections = detect_sources(direct_bg, threshold_sigma=detection_sigma)
    print(f"    Raw detections: {len(raw_detections)}")

    # ── 4. Remove out-of-field contaminants ───────────────────────────────────
    print("\n[4] Filtering contaminating traces …")
    clean_detections = filter_out_of_field_sources(
        raw_detections,
        disp_shape=dispersed_bg.shape,
        direct_shape=direct_bg.shape,
        dispersion_px_per_nm=dispersion_px_per_nm,
        lambda_min_nm=lambda_min_nm,
        lambda_max_nm=lambda_max_nm,
        lambda_ref_nm=lambda_ref_nm,
        tilt_deg=tilt_deg,
    )
    contamination_mask = build_contamination_mask(
        disp_shape=dispersed_bg.shape,
        direct_shape=direct_bg.shape,
        dispersion_px_per_nm=dispersion_px_per_nm,
        lambda_min_nm=lambda_min_nm,
        lambda_max_nm=lambda_max_nm,
        lambda_ref_nm=lambda_ref_nm,
        tilt_deg=tilt_deg,
    )
    print(f"    Sources after filtering : {len(clean_detections)}")
    print(f"    Contaminated pixels     : {contamination_mask.mean()*100:.1f}%")

    # ── 5. Gaussian PSF fitting ───────────────────────────────────────────────
    print("\n[5] Gaussian PSF fitting …")
    sources = fit_gaussian_psf(direct_bg, clean_detections, fit_box=fit_box)
    for s in sources:
        print(f"    src {s['id']:3d}  x={s['x']:7.2f}  y={s['y']:7.2f}  "
              f"A={s['amplitude']:8.1f}  σx={s['sigma_x']:.2f}  σy={s['sigma_y']:.2f}")

    # ── 6. Build a_tilde & support mask from real detections ─────────────────
    print("\n[6] Building coefficient vector a_tilde from detections …")
    a_tilde = build_a_tilde_from_sources(
        sources, direct_bg, IMAGE_SHAPE, basis, radius_factor=radius_factor
    )
    support_mask = a_tilde != 0
    print(f"    Active coefficients: {support_mask.sum()} / {len(support_mask)}")

    # ── 7. Mixing matrix (optional, same as optimized mock) ───────────────────
    M = None
    if use_mixing:
        print("\n[7] Building mixing matrix M …")
        M = build_mixing_matrix(sources, IMAGE_SHAPE, basis, radius_factor=radius_factor)
        print(f"    M shape: {M.shape}")

    # ── 8. Solve  H * a = dispersed ──────────────────────────────────────────
    print("\n[8] Running SpectralSolver …")

    # Zero out contaminated pixels so they don't bias the solve
    dispersed_clean = dispersed_bg.copy()
    dispersed_clean[contamination_mask] = 0.0

    solver    = SpectralSolver(op, max_iter=max_iter, tolerance=tolerance)
    recovered = solver.solve(dispersed_clean, support_mask=support_mask, M=M)
    print(f"    Recovered shape: {recovered.shape}")

    # ── 9. Diagnostics ────────────────────────────────────────────────────────
    H_px, W_px = IMAGE_SHAPE
    n = basis.n_components
    n_pix = H_px * W_px

    recovered_img = basis.broadband_image(recovered, IMAGE_SHAPE)
    residual_img  = np.abs(direct_bg - recovered_img)

    if plots:
        vmin_dr, vmax_dr = _clip(direct_bg)
        vmin_d2, vmax_d2 = _clip(dispersed_clean)
        vmax_res = np.nanmean(residual_img) + np.nanstd(residual_img)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        kw = dict(origin="lower", aspect="auto", interpolation="nearest", cmap="inferno")
        ims = [
            axes[0].imshow(direct_bg,       vmin=vmin_dr, vmax=vmax_dr, **kw),
            axes[1].imshow(dispersed_clean, vmin=vmin_d2, vmax=vmax_d2, **kw),
            axes[2].imshow(recovered_img,   vmin=vmin_dr, vmax=vmax_dr, **kw),
            axes[3].imshow(residual_img,    vmin=0,       vmax=vmax_res, **kw),
        ]
        for ax, im, t in zip(axes, ims,
                              ["Direct (bg-sub)", "Dispersed (masked)",
                               "Recovered", "|Residual|"]):
            ax.set_title(t); ax.set_xlabel("col"); ax.set_ylabel("row")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Overlay detections on direct image
        for s in sources:
            axes[0].plot(s['x'], s['y'], 'r+', ms=8, mew=1.5)

        fig.suptitle("Real-image recovery — spectrex", y=1.01)
        fig.tight_layout()
        plt.show()

    if parity:
        active_indices = [
            k for k in range(n_pix)
            if np.any(a_tilde[k*n:(k+1)*n] != 0)
        ]
        true_flux = np.concatenate(
            [basis.reconstruct(a_tilde[k*n:(k+1)*n]) for k in active_indices]
        )
        rec_flux = np.concatenate(
            [basis.reconstruct(recovered[k*n:(k+1)*n]) for k in active_indices]
        )

        fig, axes = plt.subplots(2, 1, figsize=(5, 8), sharex=True,
                                  height_ratios=(1, 0.6),
                                  gridspec_kw={'hspace': 0})
        outliers = rec_flux <= 1.0
        minv = max(min(true_flux.min(), rec_flux.min()), -10_000)
        maxv = max(true_flux.max(), rec_flux.max())

        axes[0].scatter(true_flux[~outliers], rec_flux[~outliers],
                        s=2, alpha=0.05, linewidths=0, color="C0", rasterized=True)
        axes[0].scatter(true_flux[outliers],  rec_flux[outliers],
                        s=2, alpha=0.05, linewidths=0, color="C1", rasterized=True)
        axes[0].plot([minv, maxv], [minv, maxv], "r--", lw=1)
        axes[0].set_aspect("equal", adjustable="box")
        axes[0].set_xlim(minv, maxv); axes[0].set_ylim(minv, maxv)
        axes[0].set_ylabel("Recovered flux f(λ)")
        axes[0].set_title("Parity plot — real images")

        frac = (true_flux - rec_flux) / (true_flux + 1e-8)
        axes[1].scatter(true_flux[~outliers], frac[~outliers],
                        s=2, alpha=0.05, linewidths=0, color="C0", rasterized=True)
        axes[1].scatter(true_flux[outliers],  frac[outliers],
                        s=2, alpha=0.05, linewidths=0, color="C1", rasterized=True)
        axes[1].set_ylim(-2, 2)
        axes[1].set_xlabel("True flux f(λ)")
        fme = np.sqrt(np.mean(frac[~outliers])**2)
        axes[1].text(0.05, 0.92, f"frac. mean error = {fme:.4f}", color="C0",
                     transform=axes[1].transAxes, fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        fig.tight_layout()
        plt.show()

    return dict(
        sources=sources,
        a_tilde=a_tilde,
        recovered=recovered,
        direct=direct_bg,
        dispersed=dispersed_clean,
        recovered_img=recovered_img,
        residual_img=residual_img,
        contamination_mask=contamination_mask,
    )


# # ── CLI ───────────────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("direct",    help="Direct image FITS file")
#     parser.add_argument("dispersed", help="Dispersed (grism) FITS file")
#     parser.add_argument("--dispersion",       type=float, default=4.65)
#     parser.add_argument("--lambda-min",       type=float, default=800.0)
#     parser.add_argument("--lambda-max",       type=float, default=1150.0)
#     parser.add_argument("--lambda-ref",       type=float, default=975.0)
#     parser.add_argument("--tilt",             type=float, default=0.0)
#     parser.add_argument("--detection-sigma",  type=float, default=5.0)
#     parser.add_argument("--fit-box",          type=int,   default=15)
#     parser.add_argument("--no-mixing",        action="store_true")
#     parser.add_argument("--parity",           action="store_true")
#     parser.add_argument("--plots",            action="store_true")
#     parser.add_argument("--cold-start",       action="store_true")
#     args = parser.parse_args()

#     run_real_image_pipeline(
#         direct_path=args.direct,
#         dispersed_path=args.dispersed,
#         dispersion_px_per_nm=args.dispersion,
#         lambda_min_nm=args.lambda_min,
#         lambda_max_nm=args.lambda_max,
#         lambda_ref_nm=args.lambda_ref,
#         tilt_deg=args.tilt,
#         detection_sigma=args.detection_sigma,
#         fit_box=args.fit_box,
#         use_mixing=not args.no_mixing,
#         parity=args.parity,
#         plots=args.plots,
#         cold_start=args.cold_start,
#     )
    

result = run_real_image_pipeline(
    direct_path=r"C:\Users\anika\GitHub\spectrex\testdata\RateFiles\Match\jw01090001001_28101_00001_nis_rate.fits",
    dispersed_path=r"C:\Users\anika\GitHub\spectrex\testdata\RateFiles\Match\jw01090001001_27101_00004_nis_rate.fits",
    plots=True,
    parity=True,
)