"""
11_register_astrometry.py — register all bands to the Gaia (NIRCam) frame
==========================================================================
SMC GO-5952 — psf_match module, step 11.

Why: NIRCam stage-3 is anchored to Gaia DR3 (tweakreg); MIRI is not
(tweakreg off — too few usable point sources), so each MIRI band
carries its own ~50-100 mas pointing error. Measured on the matched
maps (step 09C), all bands show systematic median offsets relative to
each other at the 30-105 mas level.

Fix (no resampling, no smoothing): for each filter, measure the median
star-centroid offset relative to the REFERENCE band (F150W, Gaia
frame) on the matched maps, convert it to a sky offset via the common
grid WCS, and CORRECT THE CRVAL of the input mosaics (*_zp) and of the
convolved mosaics (convolution does not change the WCS). Then rerun
04_reproject_common.py: pixels are unchanged, only their sky address
is fixed. Offsets collapse to ~0 by construction — verified by
re-running step 09C afterwards.

Outputs: psf_match/astrometric_offsets.ecsv (measured offsets, applied
         corrections), corrected WCS in data/*_zp.fits and
         psf_matching/convolved/*.fits (keywords ASTCORR1/2 + HISTORY)
Usage: conda activate jwst && python 11_register_astrometry.py [--measure-only]
"""

import sys
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.wcs import WCS
from scipy import ndimage

from config import ALL_FILTERS, ROOT, TARGET, CONV_DIR, i2d_path

REF = "F150W"                    # Gaia-anchored reference band
MATCHED = ROOT / "products" / "matched"
PIX = 0.11
OUT_TABLE = ROOT / "pipeline" / "psf_match" / "astrometric_offsets.ecsv"

maps = {f: fits.getdata(MATCHED / f"{f}_matched{TARGET}.fits", "SCI")
        for f in ALL_FILTERS}
grid_wcs = WCS(fits.getheader(MATCHED / f"{REF}_matched{TARGET}.fits", "SCI"))
ny, nx = maps[REF].shape

# ---- star census (same recipe as 09C) ----
det = maps["F200W"]
valid_all = np.all([np.isfinite(maps[f]) for f in ALL_FILTERS], axis=0)
_, med, std = sigma_clipped_stats(det[np.isfinite(det)], sigma=3)
d = np.nan_to_num(det, nan=-1e9)
peak = (d == ndimage.maximum_filter(d, size=25)) & (d > med + 15 * std)
ys, xs = np.where(peak)
cand = sorted([(det[y, x], x, y) for y, x in zip(ys, xs)
               if 40 < x < nx - 40 and 40 < y < ny - 40
               and valid_all[y - 10:y + 11, x - 10:x + 11].all()],
              reverse=True)[:30]


def centroid(data, x0, y0, box=10):
    cut = data[y0 - box:y0 + box + 1, x0 - box:x0 + box + 1].astype(float)
    cut = cut - np.nanmedian(cut)
    cut[~np.isfinite(cut)] = 0
    cut[cut < 0] = 0
    yy, xx = np.mgrid[y0 - box:y0 + box + 1, x0 - box:x0 + box + 1]
    s = cut.sum()
    return ((xx * cut).sum() / s, (yy * cut).sum() / s) if s > 0 else (np.nan,) * 2


cents = {f: [] for f in ALL_FILTERS}
for _, x, y in cand:
    cxy = {}
    for f in ALL_FILTERS:
        cx, cy = centroid(maps[f], x, y)
        if not np.isfinite(cx) or abs(cx - x) > 3 or abs(cy - y) > 3:
            cxy = None
            break
        cxy[f] = (cx, cy)
    if cxy:
        for f in ALL_FILTERS:
            cents[f].append(cxy[f])
ref_c = np.array(cents[REF])
print(f"{len(ref_c)} stars; offsets vs {REF} (Gaia frame):")

rows = []
for f in ALL_FILTERS:
    off = np.array(cents[f]) - ref_c                       # px on the grid
    dx, dy = np.median(off[:, 0]), np.median(off[:, 1])
    ex = 1.4826 * np.median(np.abs(off[:, 0] - dx)) / np.sqrt(len(off))
    ey = 1.4826 * np.median(np.abs(off[:, 1] - dy)) / np.sqrt(len(off))
    # sky correction: star measured at +d, its true position is at -d
    x0, y0 = nx / 2, ny / 2
    ra_m, dec_m = grid_wcs.pixel_to_world_values(x0 + dx, y0 + dy)
    ra_t, dec_t = grid_wcs.pixel_to_world_values(x0, y0)
    dra, ddec = ra_t - ra_m, dec_t - dec_m                 # deg, to ADD to CRVAL
    rows.append([f, dx * PIX * 1e3, dy * PIX * 1e3,
                 ex * PIX * 1e3, ey * PIX * 1e3, dra, ddec])
    print(f"  {f:8s} dx {dx * PIX * 1e3:+7.1f} +/- {ex * PIX * 1e3:4.1f} mas   "
          f"dy {dy * PIX * 1e3:+7.1f} +/- {ey * PIX * 1e3:4.1f} mas")

tab = Table(rows=rows, names=["filter", "dx_mas", "dy_mas", "err_x_mas",
                              "err_y_mas", "dCRVAL1_deg", "dCRVAL2_deg"])
tab.meta["comment"] = [
    f"Median star offsets vs {REF} (Gaia frame) measured on the matched",
    "maps; dCRVAL to ADD to the input-mosaic CRVAL to register the band.",
    f"{len(ref_c)} stars, MAD-based errors."]
tab.write(OUT_TABLE, format="ascii.ecsv", overwrite=True)
print(f"-> {OUT_TABLE}")

if "--measure-only" in sys.argv:
    sys.exit(0)

# ---- apply CRVAL corrections ----
print("\napplying CRVAL corrections (zp mosaics + convolved mosaics):")
for row in tab:
    f = row["filter"]
    dra, ddec = float(row["dCRVAL1_deg"]), float(row["dCRVAL2_deg"])
    targets = [i2d_path(f)]
    if f != TARGET:
        targets.append(CONV_DIR / f"{f}_conv_{TARGET}.fits")
    for path in targets:
        with fits.open(path, mode="update") as h:
            for ext in ("SCI", "ERR"):
                if ext in h and "CRVAL1" in h[ext].header:
                    if h[ext].header.get("ASTCORR", False):
                        raise RuntimeError(f"{path} already corrected!")
                    h[ext].header["CRVAL1"] += dra
                    h[ext].header["CRVAL2"] += ddec
                    h[ext].header["ASTCORR"] = (True, "CRVAL registered to Gaia frame")
                    h[ext].header["ASTCORR1"] = (dra, "[deg] dCRVAL1 applied (11_register)")
                    h[ext].header["ASTCORR2"] = (ddec, "[deg] dCRVAL2 applied (11_register)")
            h[0].header["HISTORY"] = ("astrometry registered to Gaia/" + REF +
                                      " frame (11_register_astrometry.py)")
    print(f"  {f:8s} dCRVAL = ({dra * 3600 * 1e3:+.1f}, {ddec * 3600 * 1e3:+.1f}) mas")
print("\nNOW RERUN: python 04_reproject_common.py   then 09 (validation)")
