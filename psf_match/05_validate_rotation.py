"""
05_validate_rotation.py — check the PSF rotation against real stars
===================================================================
The rotation applied in 01_make_psfs.py (grid_pa - PA_APER) is convention-
sensitive; this script verifies it empirically on the mosaics.

For a given filter:
  1. find bright, isolated, unsaturated stars in the mosaic (DAOStarFinder)
  2. cut out each star and measure the diffraction-spike orientation from the
     phase of the m=6 azimuthal Fourier component (measure_spike_angle)
  3. same measurement on the rotated model PSF
  4. report the offset (deg, mod 60): |offset| < ~2 deg -> rotation correct
  5. save side-by-side figures to figures/rotation_check/

Usage:
    python 05_validate_rotation.py F200W F770W
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.visualization import simple_norm
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from photutils.detection import DAOStarFinder

from config import FIG_DIR, PSF_DIR, i2d_path
from psfmatch_utils import measure_spike_angle
from regrid import regrid


def find_stars(sci, fwhm_px, nstars=3, margin=100):
    """Bright, isolated, unsaturated stars, away from the mosaic edges."""
    mean, med, std = sigma_clipped_stats(sci, sigma=3.0, maxiters=3)
    finder = DAOStarFinder(fwhm=fwhm_px, threshold=200 * std, exclude_border=True)
    tbl = finder(np.nan_to_num(sci - med))
    if tbl is None or len(tbl) == 0:
        raise RuntimeError("no stars found — lower the threshold")
    tbl.sort("flux")
    tbl.reverse()

    ny, nx = sci.shape
    picked = []
    for row in tbl:
        x, y = row["xcentroid"], row["ycentroid"]
        if not (margin < x < nx - margin and margin < y < ny - margin):
            continue
        cut = sci[int(y) - 8:int(y) + 9, int(x) - 8:int(x) + 9]
        if not np.all(np.isfinite(cut)):  # saturated cores are NaN
            continue
        if any(np.hypot(x - px, y - py) < 60 for px, py, _ in picked):
            continue
        picked.append((x, y, row["flux"]))
        if len(picked) == nstars:
            break
    return picked


def validate(filt, nstars=3):
    figdir = FIG_DIR / "rotation_check"
    figdir.mkdir(parents=True, exist_ok=True)

    with fits.open(i2d_path(filt)) as hdul:
        sci = hdul["SCI"].data.astype(float)
        pix_img = proj_plane_pixel_scales(WCS(hdul["SCI"].header))[1] * 3600.0

    with fits.open(PSF_DIR / f"{filt}_stpsf_rot.fits") as hdul:
        psf = hdul[0].data.astype(float)
        pix_psf = hdul[0].header["PIXELSCL"]
        rotang = hdul[0].header["ROTANG"]

    # PSF at image pixel scale, same annulus in arcsec for both measurements
    psf_img = regrid(psf, pix_psf, pix_img)
    r_in_as, r_out_as = 0.35, 2.0
    ang_psf = measure_spike_angle(psf_img, r_in=r_in_as / pix_img,
                                  r_out=r_out_as / pix_img, recenter=True)

    half = int(r_out_as * 1.5 / pix_img)
    stars = find_stars(sci, fwhm_px=max(2.5, 0.15 / pix_img), nstars=nstars,
                       margin=half + 10)
    print(f"{filt}: ROTANG={rotang:+.2f}  PSF spike angle={ang_psf:.2f} (mod 60)")

    fig, axes = plt.subplots(1, len(stars) + 1, figsize=(4 * (len(stars) + 1), 4))
    offsets = []
    for ax, (x, y, flux) in zip(axes, stars):
        cut = sci[int(y) - half:int(y) + half + 1, int(x) - half:int(x) + half + 1]
        ang_star = measure_spike_angle(cut, r_in=r_in_as / pix_img,
                                       r_out=r_out_as / pix_img)
        off = (ang_star - ang_psf + 30) % 60 - 30
        offsets.append(off)
        print(f"  star ({x:7.1f},{y:7.1f}): spike angle={ang_star:5.2f}  "
              f"offset={off:+5.2f} deg")
        ax.imshow(cut, norm=simple_norm(cut, "log", percent=99.9), origin="lower")
        ax.set_title(f"star  $\\Delta$={off:+.1f}$^\\circ$")

    n = len(psf_img)
    c = n // 2
    cutp = psf_img[c - half:c + half + 1, c - half:c + half + 1]
    axes[-1].imshow(cutp, norm=simple_norm(cutp, "log", percent=99.99),
                    origin="lower")
    axes[-1].set_title(f"{filt} model PSF (rotated)")
    fig.suptitle(f"{filt}: spike-angle offset = "
                 f"{np.mean(offsets):+.2f} deg (mod 60)")
    fig.tight_layout()
    fig.savefig(figdir / f"{filt}_rotation_check.png", dpi=150)
    plt.close(fig)
    return np.mean(offsets)


if __name__ == "__main__":
    filters = sys.argv[1:] or ["F200W", "F770W"]
    for filt in filters:
        try:
            validate(filt)
        except Exception as e:
            print(f"{filt}: validation failed — {e}")
