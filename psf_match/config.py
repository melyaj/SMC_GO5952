"""
config.py — PSF matching configuration for SMC GO-5952
======================================================
Common PSF (F2100W) homogenization of the 14 NIRCam+MIRI mosaics.

Method: Aniano+ 2011 kernel generation, adapted from Liz Tarantino's
sextansA_PAH_paper psf-match code (Tarantino et al. 2025).

Workflow:
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


def i2d_path(filt):
    """Final mosaic used for analysis.

    MIRI: 2D-polynomial sky-subtracted mosaics.
    NIRCam: plain stage3 mosaics (no polynomial skysub — see pipeline README).
    """
    if filt in MIRI_FILTERS:
        return ROOT / "miri" / filt / "stage3" / f"miri_{filt}_final_i2d_skysub.fits"
    return ROOT / "nircam" / filt / "stage3" / f"nircam_{filt}_final_i2d.fits"


# ---------------------------------------------------------------- products
PROD_DIR = ROOT / "psf_match"
PSF_DIR = PROD_DIR / "psfs"
KER_DIR = PROD_DIR / "kernels"
CONV_DIR = PROD_DIR / "convolved"
FIG_DIR = PROD_DIR / "figures"
ANALYSIS_DIR = ROOT / "analysis_ready"

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
