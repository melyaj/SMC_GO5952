"""
config.py — PSF matching configuration for SMC GO-5952
======================================================
Common PSF (F2100W) homogenization of the 14 NIRCam+MIRI mosaics.

Method: Aniano+ 2011 kernel generation, adapted from Liz Tarantino's
sextansA_PAH_paper psf-match code (Tarantino et al. 2025).

Workflow (canonical order — ALL background handling before PSF matching):
    00_subtract_zeropoint.py — per-filter zero-point constants at mosaic level
    01_make_psfs.py         — STPSF PSFs matched to each i2d (oversample 3, odd parity)
    02_make_kernels.py      — Aniano kernels {filter} -> F2100W + D/W- diagnostics
    03_convolve_images.py   — convolve SCI+ERR at native pixel scale
    04_reproject_common.py  — reproject to the common 0.11"/px north-up grid
    05_validate_rotation.py — check PSF rotation against real stars (spike angles)
"""
from pathlib import Path

ROOT = Path("/Users/melyajou/SMC_GO5952")

# ---------------------------------------------------------------- filters
MIRI_FILTERS = ["F560W", "F770W", "F1000W", "F1130W", "F1500W", "F2100W"]
NIRCAM_FILTERS = ["F150W", "F187N", "F200W", "F212N", "F300M", "F335M", "F360M", "F444W"]
ALL_FILTERS = NIRCAM_FILTERS + MIRI_FILTERS

TARGET = "F2100W"  # broadest PSF (FWHM ~ 0.67")


def instrument(filt):
    return "miri" if filt in MIRI_FILTERS else "nircam"


def i2d_path_raw(filt):
    """Stage-3 mosaic BEFORE zero-point (input of 00_subtract_zeropoint).

    MIRI: 2D-polynomial sky-subtracted mosaics.
    NIRCam: plain stage3 mosaics (no polynomial skysub — see pipeline README).
    """
    if filt in MIRI_FILTERS:
        return ROOT / "miri" / filt / "stage3" / f"miri_{filt}_final_i2d_skysub.fits"
    return ROOT / "nircam" / filt / "stage3" / f"nircam_{filt}_final_i2d.fits"


def i2d_path(filt):
    """Final mosaic used for PSF matching: zero-point homogenized (00)."""
    p = i2d_path_raw(filt)
    return p.with_name(p.name.replace(".fits", "_zp.fits"))


# ------------------------------------------------------- zero-point table
# measured on the matched maps (common dark cavity), applied at mosaic
# level by 00_subtract_zeropoint.py — constants commute with the chain
ZP_TABLE = Path(__file__).parent / "zeropoint_offsets.ecsv"

# ---------------------------------------------------------------- products
PROD_DIR = ROOT / "psf_match"
PSF_DIR = PROD_DIR / "psfs"
KER_DIR = PROD_DIR / "kernels"
CONV_DIR = PROD_DIR / "convolved"
FIG_DIR = PROD_DIR / "figures"
# single clean products tree: products/matched = THE science base
ANALYSIS_DIR = ROOT / "products" / "matched"

# ---------------------------------------------------------------- PSF simulation
# oversample 3 = Aniano-optimized sampling (cf. Tarantino psf.py)
OVERSAMPLE = 3
# PSF field of view: must cover the F2100W wings for kernel support
FOV_ARCSEC = {"miri": 41.25, "nircam": 30.0}

# ---------------------------------------------------------------- kernel
KAPPA = 1.0  # Aniano low-pass cutoff scaling (Tarantino default)

# ---------------------------------------------------------------- common grid
# 0.11"/px north-up TAN grid, 1244x1244 (same grid as reprojected/ composite)
GRID_REF = ROOT / "reprojected" / "F2100W_final.fits"
