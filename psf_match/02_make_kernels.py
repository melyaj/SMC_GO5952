"""
02_make_kernels.py — Aniano+ 2011 kernels {filter} -> F2100W
============================================================
Adapted from Liz Tarantino's psf_match.py (sextansA_PAH_paper); the frequency
loop is vectorized (psfmatch_utils.aniano_filter) but the method is identical.

For each source filter:
  1. load the source PSF rotated to its own mosaic grid ({filt}_stpsf_rot.fits)
  2. rotate the RAW F2100W PSF to the SOURCE mosaic grid
     (angle = grid_pa_source - PA_APER_F2100W — the MIRI and NIRCam drizzle
     grids have different orientations)
  3. regrid the F2100W PSF to the source PSF pixel scale (spline)
  4. match grid sizes, normalize, build the Aniano kernel (kappa from config)
  5. diagnostics: D, W-, radial profiles -> figures/{filt}_to_F2100W/
  6. save kernels/{filt}_to_F2100W_kernel.fits (PIXELSCL = source PSF scale)

Usage:
    python 02_make_kernels.py            # all filters except F2100W
    python 02_make_kernels.py F770W
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.visualization import simple_norm
from scipy.ndimage import rotate

from config import ALL_FILTERS, FIG_DIR, KAPPA, KER_DIR, PSF_DIR, TARGET, i2d_path
from fwhm import fwhm as fit_fwhm
from psfmatch_utils import aniano_kernel, grid_pa_y, resize_center
from radial_data import radial_data
from regrid import regrid


def make_kernel(filt, kappa=KAPPA):
    figdir = FIG_DIR / f"{filt}_to_{TARGET}"
    figdir.mkdir(parents=True, exist_ok=True)

    # -- source PSF, already in its own mosaic frame
    with fits.open(PSF_DIR / f"{filt}_stpsf_rot.fits") as hdul:
        psf1 = hdul[0].data.astype(float)
        head1 = hdul[0].header.copy()
    pix_size = head1["PIXELSCL"]

    # -- target PSF rotated to the SOURCE mosaic grid
    with fits.open(PSF_DIR / f"{TARGET}_stpsf_raw.fits") as hdul:
        psf2 = hdul[0].data.astype(float)
        head2 = hdul[0].header
    pa_aper_t = fits.getheader(i2d_path(TARGET), "SCI")["PA_APER"]
    grid_pa_s = grid_pa_y(fits.getheader(i2d_path(filt), "SCI"))
    ang = grid_pa_s - pa_aper_t
    psf2 = rotate(psf2, ang, reshape=False, prefilter=False)

    # -- common grid: regrid target to source scale, pad to the larger size
    psf2 = regrid(psf2, head2["PIXELSCL"], pix_size)
    grid_size = max(len(psf1), len(psf2))
    psf1 = resize_center(psf1, grid_size)
    psf2 = resize_center(psf2, grid_size)

    psf1 = psf1 / np.sum(psf1)
    psf2 = psf2 / np.sum(psf2)

    # -- source FWHM (2D Gaussian fit, Tarantino fwhm.py) sets the filter cutoff
    width1 = fit_fwhm(psf1, head1)
    fwhm1 = width1["xarcsec"]

    ker, diag = aniano_kernel(psf1, psf2, pix_size, fwhm1, kappa=kappa)
    conv = diag.pop("conv_check")
    print(f"{filt} -> {TARGET}: FWHM1={fwhm1:.3f}\" kappa={kappa} "
          f"D={diag['D']:.5f} W-={diag['W-']:.5f}")

    # -------------------------------------------------------------- figures
    norm = simple_norm(psf2, "log", min_cut=0, max_cut=0.01 * np.nanmax(psf1))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, im, title in zip(axes, [psf1, psf2, ker],
                             [f"{filt} PSF", f"{TARGET} PSF (regridded)", "kernel"]):
        knorm = norm if title != "kernel" else simple_norm(
            ker, "linear", min_cut=-0.1 * ker.max(), max_cut=0.1 * ker.max())
        ax.imshow(im, norm=knorm, origin="lower")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(figdir / "psfs_kernel.pdf")
    plt.close(fig)

    # radial profiles: convolved source vs target (Aniano-style check)
    psf2_rad = radial_data(psf2)
    conv_rad = radial_data(conv)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(psf2_rad.r * pix_size, psf2_rad.median, label=f"{TARGET} PSF")
    ax1.plot(conv_rad.r * pix_size, conv_rad.median, ls="--",
             label=f"{filt} $\\otimes$ kernel")
    ax1.set_xlim(0, 2)
    ax1.set_xlabel(r"$\Theta$ (arcsec)")
    ax1.set_ylabel(r"PSF ($\Psi(\Theta)$)")
    ax1.legend()

    rad1 = psf2_rad.r * psf2_rad.median
    rad2 = conv_rad.r * conv_rad.median
    ax2.plot(psf2_rad.r * pix_size, rad1 / np.nanmax(rad1), label=f"{TARGET} PSF")
    ax2.plot(conv_rad.r * pix_size, rad2 / np.nanmax(rad2), ls="--",
             label=f"{filt} $\\otimes$ kernel")
    ax2.set_xlim(0, 2)
    ax2.set_xlabel(r"$\Theta$ (arcsec)")
    ax2.set_ylabel(r"$\Theta\,\Psi(\Theta)$ / max")
    ax2.text(0.62, 0.72, f"kappa = {kappa:4.3f}\nD = {diag['D']:8.6f}\n"
             f"W- = {diag['W-']:8.6f}", transform=ax2.transAxes)
    ax2.legend()
    fig.tight_layout()
    fig.savefig(figdir / "aniano_profiles.pdf")
    plt.close(fig)

    # -------------------------------------------------------------- save
    head1["NAXIS1"] = len(ker)
    head1["NAXIS2"] = len(ker)
    head1["KAPPA"] = (kappa, "Aniano low-pass cutoff scaling")
    head1["ANIANO_D"] = (diag["D"], "sum |psf1 x ker - psf2|")
    head1["ANIANO_W"] = (diag["W-"], "sum of negative kernel values")
    head1["TARGPSF"] = (TARGET, "target PSF")
    out = KER_DIR / f"{filt}_to_{TARGET}_kernel.fits"
    fits.writeto(out, ker, head1, overwrite=True)
    print(f"  -> {out.name}")
    return diag


if __name__ == "__main__":
    filters = sys.argv[1:] or [f for f in ALL_FILTERS if f != TARGET]
    KER_DIR.mkdir(parents=True, exist_ok=True)
    for filt in filters:
        make_kernel(filt)
