from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix
from matplotlib.patches import Rectangle

import astropy.units as u

from astropy.coordinates import SkyCoord
from astropy.time import Time
from astroquery.mast import MastMissions, Observations
import numpy as np


from astropy.io import fits
from astropy.table import Table


from jwst import datamodels
from jwst.assign_wcs import AssignWcsStep

import stpsf


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
IMAGES = HERE / "unittests" / "Images"

# ── Configuration ─────────────────────────────────────────────────────────────
# set cold_start true if anything in this configuration section is changed or delete operator chache
COLD_START = False    # set True to force operator rebuild from scratch
IMAGE_SHAPE = (500, 20) # Main frame
DETECTOR_SHAPE = (355+500+10,26) # Extended frame
SOURCE_ORIGIN = (180,3) # (0,0) of Detector starts at (10,200)
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



# -------------------------------------------------------------------------
# STPSF sigma estimation
# -------------------------------------------------------------------------

def estimate_sigma_from_stpsf(
    instrument_name="NIRISS",
    filter_name="F150W",
    detector=None,
    fov_pixels=64,
    oversample=4,
):
    inst = getattr(stpsf, instrument_name)()
    inst.filter = filter_name

    if detector is not None:
        inst.detector = detector

    psf_hdul = inst.calc_psf(
        fov_pixels=fov_pixels,
        oversample=oversample,
    )

    psf = np.asarray(psf_hdul[0].data, dtype=float)

    psf = np.nan_to_num(psf, nan=0.0, posinf=0.0, neginf=0.0)

    if psf.sum() <= 0:
        raise ValueError("STPSF returned an empty or invalid PSF.")

    psf /= psf.sum()

    ny, nx = psf.shape
    YY, XX = np.mgrid[:ny, :nx]

    x0 = np.sum(XX * psf)
    y0 = np.sum(YY * psf)

    var_x = np.sum((XX - x0) ** 2 * psf)
    var_y = np.sum((YY - y0) ** 2 * psf)

    sigma_oversampled = np.sqrt(0.5 * (var_x + var_y))
    sigma_detector_pixels = sigma_oversampled / oversample

    return sigma_detector_pixels, psf


# -------------------------------------------------------------------------
# FITS helpers
# -------------------------------------------------------------------------

def read_fits_image(filename, ext="SCI"):
    with fits.open(filename) as hdul:
        if ext in hdul:
            image = hdul[ext].data
        else:
            image = hdul[0].data

    image = np.asarray(image, dtype=float)
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

    return image


def get_wcs_world_to_detector(filename):
    dm = datamodels.open(filename)
    dm = AssignWcsStep.call(dm)
    return dm.meta.wcs.get_transform("world", "detector")


# -------------------------------------------------------------------------
# Catalog reading
# -------------------------------------------------------------------------

def read_catalog_sources_world_to_detector(
    catalog_files,
    wcs,
    ra_col="RA",
    dec_col="DEC",
):
    xs = []
    ys = []
    source_catalog = []
    source_index = []

    for catalog in catalog_files:
        tab = Table.read(catalog)

        # Be permissive about column names.
        names_lower = {name.lower(): name for name in tab.colnames}

        if ra_col not in tab.colnames:
            ra_name = names_lower.get(ra_col.lower(), None)
        else:
            ra_name = ra_col

        if dec_col not in tab.colnames:
            dec_name = names_lower.get(dec_col.lower(), None)
        else:
            dec_name = dec_col

        if ra_name is None or dec_name is None:
            raise KeyError(
                f"Could not find RA/DEC columns in {catalog}. "
                f"Available columns are: {tab.colnames}"
            )

        ra = np.asarray(tab[ra_name], dtype=float)
        dec = np.asarray(tab[dec_name], dtype=float)

        x_det, y_det, *_ = wcs(ra, dec, ra, dec)

        for i, (x0, y0) in enumerate(zip(x_det, y_det)):
            if not np.isfinite(x0) or not np.isfinite(y0):
                continue

            xs.append(float(x0))
            ys.append(float(y0))
            source_catalog.append(catalog)
            source_index.append(i)

    return np.asarray(xs), np.asarray(ys), source_catalog, source_index


# -------------------------------------------------------------------------
# Gaussian support construction
# -------------------------------------------------------------------------

def estimate_local_flux(
    direct,
    x0,
    y0,
    background_radius=6,
    aperture_radius=1,
):
    """
    Estimate local source amplitude from the direct image.

    This returns a positive local peak estimate:
        local_flux = max(local_peak - local_background, 0)

    For very crowded images, replace this by a more careful photometry
    estimate later.
    """
    H, W = direct.shape

    xc = int(round(x0))
    yc = int(round(y0))

    if not (0 <= xc < W and 0 <= yc < H):
        return 0.0

    r_bg = int(background_radius)
    y_min = max(0, yc - r_bg)
    y_max = min(H, yc + r_bg + 1)
    x_min = max(0, xc - r_bg)
    x_max = min(W, xc + r_bg + 1)

    patch = direct[y_min:y_max, x_min:x_max]

    if patch.size == 0:
        return 0.0

    background = np.nanmedian(patch)

    r_ap = int(aperture_radius)
    y0_ap = max(0, yc - r_ap)
    y1_ap = min(H, yc + r_ap + 1)
    x0_ap = max(0, xc - r_ap)
    x1_ap = min(W, xc + r_ap + 1)

    aperture = direct[y0_ap:y1_ap, x0_ap:x1_ap]

    if aperture.size == 0:
        return 0.0

    peak = np.nanmax(aperture)

    return max(float(peak - background), 0.0)


def estimate_noise_level(
    direct,
    method="mad",
    fixed_noise=None,
):
    if fixed_noise is not None:
        return float(fixed_noise)

    data = np.asarray(direct, dtype=float)
    data = data[np.isfinite(data)]

    if data.size == 0:
        raise ValueError("Cannot estimate noise from empty image.")

    if method == "mad":
        med = np.median(data)
        mad = np.median(np.abs(data - med))
        noise = 1.4826 * mad
    elif method == "std":
        noise = np.std(data)
    else:
        raise ValueError("method must be 'mad' or 'std'.")

    return max(float(noise), 1e-12)


def radius_from_flux_threshold(
    local_flux,
    sigma,
    noise_level,
    noise_factor=3.0,
    min_radius_sigma=1.5,
    max_radius_sigma=6.0,
):
    """
    Choose radius from:

        local_flux * exp(-r^2 / (2 sigma^2)) >= threshold

    with threshold = noise_factor * noise_level.

    Hence:

        r = sigma * sqrt(2 log(local_flux / threshold))

    If local_flux <= threshold, use a small fallback radius.
    """
    threshold = noise_factor * noise_level

    min_radius = min_radius_sigma * sigma
    max_radius = max_radius_sigma * sigma

    if local_flux <= threshold or local_flux <= 0:
        return min_radius

    radius = sigma * np.sqrt(2.0 * np.log(local_flux / threshold))

    return float(np.clip(radius, min_radius, max_radius))


def gaussian_source_pixels_from_direct_image(
    direct,
    x0,
    y0,
    sigma,
    noise_level,
    noise_factor=3.0,
    min_radius_sigma=1.5,
    max_radius_sigma=6.0,
    background_radius=6,
    aperture_radius=1,
):
    H, W = direct.shape

    local_flux = estimate_local_flux(
        direct,
        x0=x0,
        y0=y0,
        background_radius=background_radius,
        aperture_radius=aperture_radius,
    )

    radius = radius_from_flux_threshold(
        local_flux=local_flux,
        sigma=sigma,
        noise_level=noise_level,
        noise_factor=noise_factor,
        min_radius_sigma=min_radius_sigma,
        max_radius_sigma=max_radius_sigma,
    )

    r = int(np.ceil(radius))

    x_min = max(0, int(np.floor(x0)) - r)
    x_max = min(W, int(np.floor(x0)) + r + 1)

    y_min = max(0, int(np.floor(y0)) - r)
    y_max = min(H, int(np.floor(y0)) + r + 1)

    YY, XX = np.mgrid[y_min:y_max, x_min:x_max]

    gauss = np.exp(
        -((XX - x0) ** 2 + (YY - y0) ** 2) / (2.0 * sigma**2)
    )

    # Peak normalization: centroid/peak has value one.
    if gauss.size > 0 and gauss.max() > 0:
        gauss /= gauss.max()

    # Keep pixels inside the radius.
    rr = np.sqrt((XX - x0) ** 2 + (YY - y0) ** 2)
    mask = rr <= radius

    pixels = (YY[mask] * W + XX[mask]).astype(int)
    amplitudes = gauss[mask].astype(float)

    return {
        "center": (y0, x0),
        "pixels": pixels,
        "amplitudes": amplitudes,
        "sigma": sigma,
        "radius": radius,
        "local_flux": local_flux,
    }


# -------------------------------------------------------------------------
# Full real-data recovery method
# -------------------------------------------------------------------------

def run_real_scene_optimized_recovery(
    direct_fits,
    dispersed_fits,
    catalog_files,
    op,
    basis,
    wcs_reference_fits=None,
    direct_ext="SCI",
    dispersed_ext="SCI",
    ra_col="RA",
    dec_col="DEC",
    solver_kwargs=None,
    instrument_name="NIRISS",
    filter_name="F150W",
    detector=None,
    stpsf_fov_pixels=64,
    stpsf_oversample=4,
    fixed_sigma=None,
    fixed_noise=None,
    noise_method="mad",
    noise_factor=3.0,
    min_radius_sigma=1.5,
    max_radius_sigma=6.0,
    background_radius=6,
    aperture_radius=1,
    PLOTS=False,
):
    """
    Real-data version of run_mock_scene_optimized_recovery.

    Main differences from the mock version:
        - direct is read from direct_fits
        - dispersed is read from dispersed_fits
        - source positions are read from catalog_files
        - source supports are Gaussian supports estimated from the direct image
        - one common sigma is estimated from STPSF unless fixed_sigma is provided
        - radius changes from source to source via local_flux / noise_level

    The unknown is now source-level spectral coefficients:

        x_src.shape = (n_sources * basis.n_components,)

    and the pixel-level coefficient image is reconstructed by:

        recovered = M @ x_src
    """

    if solver_kwargs is None:
        solver_kwargs = dict(max_iter=500, tolerance=1e-8)

    # ---------------------------------------------------------------------
    # Images
    # ---------------------------------------------------------------------

    direct = read_fits_image(direct_fits, ext=direct_ext)
    dispersed = read_fits_image(dispersed_fits, ext=dispersed_ext)

    DETECTOR_SHAPE = direct.shape
    IMAGE_SHAPE = dispersed.shape

    H, W = DETECTOR_SHAPE
    K, L = IMAGE_SHAPE

    n_pix_det = H * W
    n = basis.n_components

    print("Direct shape:", direct.shape)
    print("Dispersed shape:", dispersed.shape)
    print("Basis components:", n)

    # ---------------------------------------------------------------------
    # WCS
    # ---------------------------------------------------------------------

    if wcs_reference_fits is None:
        wcs_reference_fits = direct_fits

    wcs = get_wcs_world_to_detector(wcs_reference_fits)

    x_det, y_det, source_catalog, source_index = read_catalog_sources_world_to_detector(
        catalog_files=catalog_files,
        wcs=wcs,
        ra_col=ra_col,
        dec_col=dec_col,
    )

    # ---------------------------------------------------------------------
    # Sigma and noise
    # ---------------------------------------------------------------------

    if fixed_sigma is None:
        sigma, psf = estimate_sigma_from_stpsf(
            instrument_name=instrument_name,
            filter_name=filter_name,
            detector=detector,
            fov_pixels=stpsf_fov_pixels,
            oversample=stpsf_oversample,
        )
    else:
        sigma = float(fixed_sigma)
        psf = None

    noise_level = estimate_noise_level(
        direct,
        method=noise_method,
        fixed_noise=fixed_noise,
    )

    print(f"Estimated sigma: {sigma:.6f} detector pixels")
    print(f"Estimated noise level: {noise_level:.6e}")
    print(f"Threshold = noise_factor * noise_level = {noise_factor * noise_level:.6e}")

    # ---------------------------------------------------------------------
    # Source creation from real catalogs
    # ---------------------------------------------------------------------

    sources = {}
    pixel_to_sources = {}

    source_id = 0

    for x0, y0, cat, idx in zip(x_det, y_det, source_catalog, source_index):

        if not np.isfinite(x0) or not np.isfinite(y0):
            continue

        if not (0 <= x0 < W and 0 <= y0 < H):
            continue

        src = gaussian_source_pixels_from_direct_image(
            direct=direct,
            x0=x0,
            y0=y0,
            sigma=sigma,
            noise_level=noise_level,
            noise_factor=noise_factor,
            min_radius_sigma=min_radius_sigma,
            max_radius_sigma=max_radius_sigma,
            background_radius=background_radius,
            aperture_radius=aperture_radius,
        )

        if len(src["pixels"]) == 0:
            continue

        src["catalog"] = cat
        src["catalog_index"] = idx

        sources[source_id] = src

        for kk in src["pixels"]:
            pixel_to_sources.setdefault(int(kk), []).append(source_id)

        source_id += 1

    n_src = len(sources)

    print(f"Catalog sources inside detector: {n_src}")

    if n_src == 0:
        raise ValueError("No catalog sources landed inside the detector.")

    # ---------------------------------------------------------------------
    # Mixing matrix M
    # ---------------------------------------------------------------------

    rows = []
    cols = []
    data = []

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
        shape=(n_pix_det * n, n_src * n),
    ).tocsr()

    print("[M] shape:", M.shape)
    print("[M] nnz:", M.nnz)

    active_pixel_blocks = np.zeros(n_pix_det, dtype=bool)

    for s in sources.values():
        active_pixel_blocks[s["pixels"]] = True

    support_mask = np.repeat(active_pixel_blocks, n)

    pixel_density = active_pixel_blocks.sum() / n_pix_det * 100.0
    source_density = n_src / n_pix_det * 100.0

    print(f"Pixel density: {pixel_density:.6f}%")
    print(f"Source density: {source_density:.6f}%")

    print(f"Dispersed image range: [{dispersed.min():.4f}, {dispersed.max():.4f}]")

    # ---------------------------------------------------------------------
    # Recovery
    # ---------------------------------------------------------------------

    solver = SpectralSolver(op, **solver_kwargs)

    # With M, the solver should solve:
    #
    #     min_x || H M x - dispersed ||_2
    #
    # If your solve method returns source coefficients x_src, we convert them
    # back to detector-pixel coefficients with M @ x_src.
    #
    # If your solve method already returns detector-pixel coefficients,
    # remove the "M @ x_src" line below.
    x_src = solver.solve(
        dispersed,
        support_mask=support_mask,
        M=M,
    )

    print("Recovered source vector shape:", x_src.shape)

    if x_src.shape[0] == n_src * n:
        recovered = M @ x_src
    elif x_src.shape[0] == n_pix_det * n:
        recovered = x_src
    else:
        raise ValueError(
            f"Unexpected recovered vector shape {x_src.shape}. "
            f"Expected either {n_src*n} or {n_pix_det*n}."
        )

    recovered = np.asarray(recovered).ravel()

    print("Recovered detector coefficient vector shape:", recovered.shape)

    # ---------------------------------------------------------------------
    # Recovered direct image and residuals
    # ---------------------------------------------------------------------

    recovered_img = basis.broadband_image(recovered, DETECTOR_SHAPE)
    residual_img = np.abs(direct - recovered_img)

    residual_dispersion = np.abs(
        dispersed - op.apply(recovered).reshape(IMAGE_SHAPE)
    )

    # ---------------------------------------------------------------------
    # Optional plots
    # ---------------------------------------------------------------------

    if PLOTS:

        def _clip(img, lo=1, hi=99):
            finite = img[np.isfinite(img)]
            if finite.size == 0:
                return 0.0, 1.0
            return np.percentile(finite, lo), np.percentile(finite, hi)

        vmin_dr, vmax_dr = _clip(direct)
        vmin_d2, vmax_d2 = _clip(dispersed)
        _, vmax_res_img = _clip(residual_img, lo=1, hi=99)
        _, vmax_res_disp = _clip(residual_dispersion, lo=1, hi=99)

        fig, axes = plt.subplots(
            1,
            4,
            figsize=(16, 4),
            constrained_layout=True,
        )

        kw = dict(
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap="inferno",
        )

        im0 = axes[0].imshow(direct, vmin=vmin_dr, vmax=vmax_dr, **kw)
        im1 = axes[1].imshow(dispersed, vmin=vmin_d2, vmax=vmax_d2, **kw)
        im2 = axes[2].imshow(recovered_img, vmin=vmin_dr, vmax=vmax_dr, **kw)
        im3 = axes[3].imshow(
            residual_dispersion,
            vmin=0,
            vmax=vmax_res_disp,
            **kw,
        )

        titles = [
            "Direct image",
            "Dispersed image",
            "Recovered direct image",
            "|dispersed - H recovered|",
        ]

        for ax, im, title in zip(axes, [im0, im1, im2, im3], titles):
            ax.set_title(title)
            ax.set_xlabel("column")
            ax.set_ylabel("row")
            fig.colorbar(im, ax=ax)

        fig.suptitle(
            f"Real-data recovery, "
            f"n_src={n_src}, "
            f"pd={pixel_density:.4f}%, "
            f"sigma={sigma:.3f}"
        )

        plt.show()

        fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
        ax.imshow(direct, vmin=vmin_dr, vmax=vmax_dr, **kw)

        for s in sources.values():
            y0, x0 = s["center"]
            radius = s["radius"]
            circ = plt.Circle(
                (x0, y0),
                radius,
                fill=False,
                edgecolor="cyan",
                linewidth=0.5,
                alpha=0.5,
            )
            ax.add_patch(circ)

        ax.set_title("Gaussian source supports")
        ax.set_xlabel("column")
        ax.set_ylabel("row")
        plt.show()

    # ---------------------------------------------------------------------
    # Return
    # ---------------------------------------------------------------------

    return {
        "recovered": recovered,
        "x_src": x_src,
        "M": M,
        "sources": sources,
        "direct": direct,
        "dispersed": dispersed,
        "recovered_img": recovered_img,
        "residual_img": residual_img,
        "residual_dispersion": residual_dispersion,
        "sigma": sigma,
        "psf": psf,
        "noise_level": noise_level,
        "pixel density": pixel_density,
        "source density": source_density,
        "support_mask": support_mask,
    }
    
# result = run_real_scene_optimized_recovery(
#     direct_fits="direct_image.fits",
#     dispersed_fits="dispersed_image.fits",
#     catalog_files=[
#         "tri-00-ir.cat.fits",
#         "tri-02-ir.cat.fits",
#     ],
#     op=op,
#     basis=basis,
#     wcs_reference_fits="direct_image.fits",
#     ra_col="RA",
#     dec_col="DEC",
#     instrument_name="NIRISS",
#     filter_name="F150W",
#     detector=None,
#     noise_factor=3.0,
#     min_radius_sigma=1.5,
#     max_radius_sigma=6.0,
#     PLOTS=True,
# )


def query_niriss_program(program=3383):
    missions = MastMissions(mission="jwst")

    obs = missions.query_criteria(
        instrume="NIRISS",
        program=program,
        select_cols=[
            "filename",
            "productLevel",
            "targprop",
            "targ_ra",
            "targ_dec",
            "instrume",
            "exp_type",
            "filter",
            "date_obs",
            "time_obs",
            "duration",
            "program",
            "opticalElements",
            "observtn",
            "visit",
            "niriss_pupil",
            "niriss_fwcpos",
            "niriss_pwcpos",
        ],
    )

    return obs


def add_time_and_position_columns(obs):
    obs["coord"] = SkyCoord(obs["targ_ra"] * u.deg, obs["targ_dec"] * u.deg)

    # Some JWST tables have date_obs and time_obs separately.
    if "time_obs" in obs.colnames:
        t = Time(
            [f"{d}T{t}" for d, t in zip(obs["date_obs"], obs["time_obs"])],
            format="isot",
            scale="utc",
        )
    else:
        t = Time(obs["date_obs"], format="isot", scale="utc")

    obs["mjd"] = t.mjd
    return obs


def find_direct_grism_pairs(
    obs,
    target_ra,
    target_dec,
    max_sep_arcsec=5.0,
    max_time_delta_min=30.0,
    grism="GR150R",
    same_filter=True,
):
    target = SkyCoord(target_ra * u.deg, target_dec * u.deg)

    coords = SkyCoord(obs["targ_ra"] * u.deg, obs["targ_dec"] * u.deg)
    sep = coords.separation(target).arcsec

    obs = obs[sep < max_sep_arcsec]
    obs["sep_arcsec"] = sep[sep < max_sep_arcsec]

    direct_mask = np.array([
        "IMAGE" in str(x).upper()
        for x in obs["exp_type"]
    ])

    grism_mask = np.array([
        ("WFSS" in str(exp).upper() or "GRISM" in str(exp).upper())
        and grism in str(pupil).upper()
        for exp, pupil in zip(obs["exp_type"], obs["niriss_pupil"])
    ])

    direct = obs[direct_mask]
    grism_obs = obs[grism_mask]

    pairs = []

    for g in grism_obs:
        candidates = []

        for d in direct:
            same_obs_block = (
                str(g["program"]) == str(d["program"])
                and str(g["observtn"]) == str(d["observtn"])
                and str(g["visit"]) == str(d["visit"])
            )

            if not same_obs_block:
                continue

            if same_filter and str(g["filter"]) != str(d["filter"]):
                continue

            dg = SkyCoord(g["targ_ra"] * u.deg, g["targ_dec"] * u.deg)
            dd = SkyCoord(d["targ_ra"] * u.deg, d["targ_dec"] * u.deg)
            pair_sep = dg.separation(dd).arcsec

            dt_min = abs(g["mjd"] - d["mjd"]) * 24.0 * 60.0

            if pair_sep <= max_sep_arcsec and dt_min <= max_time_delta_min:
                candidates.append((dt_min, pair_sep, d, g))

        if candidates:
            candidates.sort(key=lambda x: (x[0], x[1]))
            pairs.append(candidates[0])

    return pairs
obs = query_niriss_program(program=3383)
obs = add_time_and_position_columns(obs)

pairs = find_direct_grism_pairs(
    obs,
    target_ra=23.35,
    target_dec=30.49,
    max_sep_arcsec=5.0,
    max_time_delta_min=30.0,
    grism="GR150R",
    same_filter=True,
)

for dt_min, sep_arcsec, direct, grism in pairs:
    print()
    print("PAIR")
    print("dt [min]   =", dt_min)
    print("sep [arcsec] =", sep_arcsec)
    print("direct:", direct["filename"], direct["exp_type"], direct["filter"], direct["niriss_pupil"])
    print("grism: ", grism["filename"], grism["exp_type"], grism["filter"], grism["niriss_pupil"])