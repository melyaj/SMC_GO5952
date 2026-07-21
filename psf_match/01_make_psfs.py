"""
01_make_psfs.py — STPSF PSFs for all 14 filters, rotated to each mosaic grid
============================================================================
Adapted from Liz Tarantino's psf.py (sextansA_PAH_paper).

For each filter:
  1. stpsf.setup_sim_to_match_file(i2d) — instrument/filter/detector/OPD by date
  2. calc_psf(fov_arcsec, oversample=3, parity odd) -> {filt}_stpsf_raw.fits
  3. rotate the oversampled undistorted PSF (ext 0) from the detector frame to
     the mosaic pixel grid: angle = grid_pa - PA_APER (Tarantino: -PA_APER for
     a north-up grid) -> {filt}_stpsf_rot.fits

Usage:
    python 01_make_psfs.py            # all filters
    python 01_make_psfs.py F770W F2100W
"""
import sys
import warnings

import numpy as np
from astropy.io import fits
from scipy.ndimage import rotate

from config import ALL_FILTERS, FOV_ARCSEC, OVERSAMPLE, PSF_DIR, i2d_path, instrument
from psfmatch_utils import grid_pa_y, psf_rot_angle


def build_sim(filt):
    """STPSF simulator matched to the i2d; falls back to a generic instrument
    setup if setup_sim_to_match_file fails (e.g. no network for OPD maps)."""
    import stpsf

    infile = str(i2d_path(filt))
    try:
        sim = stpsf.setup_sim_to_match_file(infile)
    except Exception as e:
        warnings.warn(f"{filt}: setup_sim_to_match_file failed ({e}); "
                      "using default OPD")
        sim = stpsf.MIRI() if instrument(filt) == "miri" else stpsf.NIRCam()
        sim.filter = filt
    sim.options["parity"] = "odd"
    return sim


def make_psf(filt, overwrite=False):
    raw_file = PSF_DIR / f"{filt}_stpsf_raw.fits"
    rot_file = PSF_DIR / f"{filt}_stpsf_rot.fits"

    sci = fits.getheader(i2d_path(filt), "SCI")
    pa_aper = sci["PA_APER"]
    grid_pa = grid_pa_y(sci)
    ang = psf_rot_angle(sci)

    if not raw_file.exists() or overwrite:
        sim = build_sim(filt)
        fov = FOV_ARCSEC[instrument(filt)]
        print(f"{filt}: calc_psf fov={fov}\" oversample={OVERSAMPLE} ...")
        sim.calc_psf(outfile=str(raw_file), fov_arcsec=fov, oversample=OVERSAMPLE)
    else:
        print(f"{filt}: raw PSF exists, skipping calc_psf")

    # rotate the oversampled, undistorted PSF (ext 0) to the mosaic frame
    with fits.open(raw_file) as hdul:
        psf = hdul[0].data.astype(float)
        head = hdul[0].header.copy()

    rot_psf = rotate(psf, ang, reshape=False, prefilter=False)
    rot_psf /= np.sum(rot_psf)

    head["PA_APER"] = (pa_aper, "mosaic aperture PA (deg E of N)")
    head["GRIDPA"] = (grid_pa, "mosaic +y axis PA (deg E of N)")
    head["ROTANG"] = (ang, "rotation applied to detector-frame PSF (deg)")
    fits.writeto(rot_file, rot_psf, head, overwrite=True)
    print(f"{filt}: PA_APER={pa_aper:.2f} grid_pa={grid_pa:.2f} "
          f"rot={ang:.2f} -> {rot_file.name}")


if __name__ == "__main__":
    filters = sys.argv[1:] or ALL_FILTERS
    PSF_DIR.mkdir(parents=True, exist_ok=True)
    for filt in filters:
        make_psf(filt)
