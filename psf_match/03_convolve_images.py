"""
03_convolve_images.py — convolve each mosaic to the F2100W PSF
==============================================================
Adapted from Liz Tarantino's convolve_image.py / convolve_err.py, with two
changes for the large NIRCam SW mosaics (7333x15529 px, 16 GB RAM):
  - the KERNEL is resampled to the image pixel scale (not the image to the
    kernel scale) — mathematically equivalent, fits in memory
  - overlap-add FFT convolution (scipy.signal.oaconvolve) with coverage
    normalization at NaN edges (Lyot cut, mosaic borders)

SCI is convolved directly; ERR is propagated as sqrt(ERR^2 (x) kernel^2) —
the exact correlation-free propagation, and positive-definite (the
var (x) kernel approximation goes negative under the Aniano negative lobes,
which produced NaN patches in ERR). Flux conservation is checked on the
interior of the field.

Usage:
    python 03_convolve_images.py            # all filters except F2100W
    python 03_convolve_images.py F770W
"""
import sys

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from config import ALL_FILTERS, CONV_DIR, KER_DIR, TARGET, i2d_path
from psfmatch_utils import (convolve_variance, convolve_with_kernel,
                            kernel_to_image_scale)


def convolve_filter(filt):
    infile = i2d_path(filt)
    with fits.open(infile) as hdul:
        sci = hdul["SCI"].data.astype(float)
        err = hdul["ERR"].data.astype(float)
        sci_head = hdul["SCI"].header.copy()

    with fits.open(KER_DIR / f"{filt}_to_{TARGET}_kernel.fits") as khdul:
        ker = khdul[0].data.astype(float)
        pix_ker = khdul[0].header["PIXELSCL"]
        kappa = khdul[0].header.get("KAPPA", 1.0)

    pix_img = proj_plane_pixel_scales(WCS(sci_head))[1] * 3600.0
    ker_img = kernel_to_image_scale(ker, pix_ker, pix_img)
    print(f"{filt}: image {sci.shape} @ {pix_img:.4f}\"/px, "
          f"kernel {ker.shape} @ {pix_ker:.4f}\" -> {ker_img.shape}")

    conv_sci = convolve_with_kernel(sci, ker_img)
    conv_err = np.sqrt(convolve_variance(err ** 2, ker_img))

    # flux conservation, away from coverage edges
    ny, nx = sci.shape
    sl = np.s_[ny // 4: 3 * ny // 4, nx // 4: 3 * nx // 4]
    f1, f2 = np.nansum(sci[sl]), np.nansum(conv_sci[sl])
    print(f"  flux conservation (interior): {100 * (f1 - f2) / f1:+.3f} %")

    sci_head["KERNEL"] = (f"{filt}_to_{TARGET}_kernel.fits", "Aniano kernel")
    sci_head["KAPPA"] = (kappa, "Aniano low-pass cutoff scaling")
    sci_head["TARGPSF"] = (TARGET, "PSF-matched to this filter")
    hdul_out = fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(conv_sci.astype(np.float32), sci_head, name="SCI"),
        fits.ImageHDU(conv_err.astype(np.float32), sci_head, name="ERR"),
    ])
    out = CONV_DIR / f"{filt}_conv_{TARGET}.fits"
    hdul_out.writeto(out, overwrite=True)
    print(f"  -> {out.name}")


if __name__ == "__main__":
    filters = sys.argv[1:] or [f for f in ALL_FILTERS if f != TARGET]
    CONV_DIR.mkdir(parents=True, exist_ok=True)
    for filt in filters:
        convolve_filter(filt)
