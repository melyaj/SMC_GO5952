"""
04_reproject_common.py — reproject the PSF-matched mosaics to the common grid
=============================================================================
Target grid: the 0.11"/px north-up TAN grid of reprojected/F2100W_final.fits
(1244x1244), so the analysis-ready maps overlay the existing composite.

reproject_adaptive (DeForest 2004) is used for SCI: flux-correct averaging for
surface brightness (MJy/sr) when downsampling the NIRCam mosaics. ERR is
propagated through the same resampling on the variance (approximate — the
drizzled noise is already correlated between pixels).

F2100W itself is not convolved (it IS the target PSF); it is only reprojected
from its native i2d.

Output: analysis_ready/{filt}_matched{TARGET}.fits  (SCI + ERR)

Usage:
    python 04_reproject_common.py            # all 14 filters
    python 04_reproject_common.py F770W F2100W
"""
import sys

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_adaptive

from config import ALL_FILTERS, ANALYSIS_DIR, CONV_DIR, GRID_REF, TARGET, i2d_path


def reproject_filter(filt, targ_head, targ_wcs, shape_out):
    if filt == TARGET:
        with fits.open(i2d_path(filt)) as hdul:
            sci = hdul["SCI"].data.astype(float)
            err = hdul["ERR"].data.astype(float)
            wcs_in = WCS(hdul["SCI"].header)
        provenance = "native i2d (target PSF, not convolved)"
    else:
        with fits.open(CONV_DIR / f"{filt}_conv_{TARGET}.fits") as hdul:
            sci = hdul["SCI"].data.astype(float)
            err = hdul["ERR"].data.astype(float)
            wcs_in = WCS(hdul["SCI"].header)
        provenance = f"convolved to {TARGET} PSF (Aniano kernel)"

    print(f"{filt}: reproject {sci.shape} -> {shape_out} ...")
    # each extension keeps its own footprint: masking SCI with the ERR
    # footprint would leak ERR NaNs into the science map
    sci_out, foot_s = reproject_adaptive((sci, wcs_in), targ_wcs,
                                         shape_out=shape_out,
                                         conserve_flux=False)
    var_out, foot_v = reproject_adaptive((err ** 2, wcs_in), targ_wcs,
                                         shape_out=shape_out,
                                         conserve_flux=False)
    err_out = np.sqrt(np.clip(var_out, 0.0, None))
    sci_out[foot_s == 0] = np.nan
    err_out[foot_v == 0] = np.nan

    head = targ_head.copy()
    head["BUNIT"] = "MJy/sr"
    head["FILTER"] = filt
    head["TARGPSF"] = (TARGET, "PSF-matched to this filter")
    # carry the zero-point keywords from the mosaic (00_subtract_zeropoint)
    with fits.open(i2d_path(filt)) as hdul_zp:
        for key in ("ZPOFF", "ZPSYS", "ASTCORR", "ASTCORR1", "ASTCORR2"):
            if key in hdul_zp["SCI"].header:
                head[key] = (hdul_zp["SCI"].header[key],
                             hdul_zp["SCI"].header.comments[key])
    head["HISTORY"] = provenance
    head["HISTORY"] = "reproject_adaptive onto reprojected/F2100W_final.fits grid"
    hdul_out = fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(sci_out.astype(np.float32), head, name="SCI"),
        fits.ImageHDU(err_out.astype(np.float32), head, name="ERR"),
    ])
    out = ANALYSIS_DIR / f"{filt}_matched{TARGET}.fits"
    hdul_out.writeto(out, overwrite=True)
    print(f"  -> {out.name}")


if __name__ == "__main__":
    filters = sys.argv[1:] or ALL_FILTERS
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    targ_head = fits.getheader(GRID_REF)
    targ_wcs = WCS(targ_head)
    shape_out = (targ_head["NAXIS2"], targ_head["NAXIS1"])
    for filt in filters:
        reproject_filter(filt, targ_head, targ_wcs, shape_out)
