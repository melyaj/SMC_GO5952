"""
12_register_v2_snr.py — astrometric registration v2, dilution-free
===================================================================
SMC GO-5952 — psf_match module, step 12. Supersedes the residual part
of step 11.

Lesson (2026-07-23): centroids of stars that are FAINT in a given band
are noise-dominated and regress toward the search-window center (= the
reference-band position), diluting measured offsets toward zero by up
to a factor ~3 (proven on F2100W: faint-star median +30 mas vs
+116 +/- 15 mas for the 8 high-SNR stars and +109 for the bright
validation star). Step-11 corrections were partially diluted.

v2: for each band, measure the median offset vs F150W (Gaia frame)
using ONLY stars with peak SNR > 8 at that band's wavelength (falling
back to >5 if fewer than 6 stars). Apply an additional CRVAL
correction when the offset is significant (|off| > 2 sigma_median),
cumulative with step 11 (ASTCORR keywords updated).

Usage: conda activate jwst && python 12_register_v2_snr.py [--measure-only]
Then rerun 04_reproject_common.py for the corrected bands.
"""

import sys
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.wcs import WCS
from scipy import ndimage

from config import ALL_FILTERS, ROOT, TARGET, CONV_DIR, i2d_path

REF = "F150W"
M = ROOT / "products" / "matched"
PIX = 0.11
OUT_TABLE = ROOT / "pipeline" / "psf_match" / "astrometric_offsets_v2.ecsv"

maps = {f: fits.getdata(M / f"{f}_matched{TARGET}.fits", "SCI")
        for f in ALL_FILTERS}
grid = WCS(fits.getheader(M / f"{REF}_matched{TARGET}.fits", "SCI"))
ny, nx = maps[REF].shape


def cen(data, x0, y0, box=10):
    c = data[y0 - box:y0 + box + 1, x0 - box:x0 + box + 1].astype(float)
    c = c - np.nanmedian(c)
    c[~np.isfinite(c)] = 0
    c[c < 0] = 0
    yy, xx = np.mgrid[y0 - box:y0 + box + 1, x0 - box:x0 + box + 1]
    s = c.sum()
    return ((xx * c).sum() / s, (yy * c).sum() / s) if s > 0 else (np.nan,) * 2


# candidate stars from the reference band
_, md, sd = sigma_clipped_stats(maps[REF][np.isfinite(maps[REF])], sigma=3)
d = np.nan_to_num(maps[REF], nan=-1e9)
pk = (d == ndimage.maximum_filter(d, size=25)) & (d > md + 15 * sd)
ys, xs = np.where(pk)
cand = [(x, y) for y, x in zip(ys, xs)
        if 40 < x < nx - 40 and 40 < y < ny - 40]
ref_cent = {(x, y): cen(maps[REF], x, y) for x, y in cand}


def band_offsets(f, snr_min):
    offs = []
    for (x, y), a in ref_cent.items():
        if not np.isfinite(a[0]):
            continue
        w = maps[f][y - 10:y + 11, x - 10:x + 11]
        if w.shape != (21, 21) or not np.isfinite(w).all():
            continue
        bg = np.nanmedian(w)
        lsd = 1.4826 * np.nanmedian(np.abs(w - bg))
        if (np.nanmax(w[7:14, 7:14]) - bg) / (lsd + 1e-9) < snr_min:
            continue
        b = cen(maps[f], x, y)
        if all(np.isfinite(v) for v in b) and abs(b[0] - a[0]) < 3 \
                and abs(b[1] - a[1]) < 3:
            offs.append((b[0] - a[0], b[1] - a[1]))
    return np.array(offs)


rows = []
print(f"{'filtre':8s} {'N*':>4s} {'SNRmin':>6s} {'dx(mas)':>9s} {'dy(mas)':>9s} "
      f"{'err':>5s}  significatif?")
for f in ALL_FILTERS:
    if f == REF:
        rows.append([f, 0, 0.0, 0.0, 0.0, False, 0.0, 0.0])
        continue
    o = band_offsets(f, 8)
    snr_used = 8
    if len(o) < 6:
        o = band_offsets(f, 5)
        snr_used = 5
    o_mas = o * PIX * 1e3
    dx, dy = np.median(o_mas[:, 0]), np.median(o_mas[:, 1])
    err = 1.4826 * np.median(np.abs(o_mas - np.median(o_mas, axis=0)),
                             axis=0).mean() / np.sqrt(len(o))
    sig = bool(np.hypot(dx, dy) > 2 * err and np.hypot(dx, dy) > 15)
    # sky correction (to ADD to CRVAL): true - measured
    ra_m, dec_m = grid.pixel_to_world_values(nx / 2 + dx / PIX / 1e3,
                                             ny / 2 + dy / PIX / 1e3)
    ra_t, dec_t = grid.pixel_to_world_values(nx / 2, ny / 2)
    rows.append([f, len(o), dx, dy, err, sig,
                 float(ra_t - ra_m), float(dec_t - dec_m)])
    print(f"{f:8s} {len(o):4d} {snr_used:6d} {dx:+9.1f} {dy:+9.1f} {err:5.1f}"
          f"  {'OUI' if sig else 'non'}")

tab = Table(rows=rows, names=["filter", "nstars", "dx_mas", "dy_mas",
                              "err_mas", "significant", "dCRVAL1_deg",
                              "dCRVAL2_deg"])
tab.meta["comment"] = [
    "v2 dilution-free offsets vs F150W (Gaia): high-SNR stars only",
    "(peak SNR>8 per band, fallback >5). Corrections applied only when",
    "|offset| > max(2 err, 15 mas). Cumulative with step 11."]
tab.write(OUT_TABLE, format="ascii.ecsv", overwrite=True)
print(f"-> {OUT_TABLE}")

if "--measure-only" in sys.argv:
    sys.exit(0)

print("\napplication des corrections significatives:")
corrected = []
for row in tab:
    if not row["significant"]:
        continue
    f = row["filter"]
    dra, ddec = float(row["dCRVAL1_deg"]), float(row["dCRVAL2_deg"])
    targets = [i2d_path(f)]
    if f != TARGET:
        targets.append(CONV_DIR / f"{f}_conv_{TARGET}.fits")
    for path in targets:
        with fits.open(path, mode="update") as h:
            for ext in ("SCI", "ERR"):
                if ext in h and "CRVAL1" in h[ext].header:
                    hdr = h[ext].header
                    hdr["CRVAL1"] += dra
                    hdr["CRVAL2"] += ddec
                    hdr["ASTCORR1"] = (hdr.get("ASTCORR1", 0.0) + dra,
                                       "[deg] cumulative dCRVAL1 (11+12)")
                    hdr["ASTCORR2"] = (hdr.get("ASTCORR2", 0.0) + ddec,
                                       "[deg] cumulative dCRVAL2 (11+12)")
            h[0].header["HISTORY"] = \
                "v2 dilution-free registration (12_register_v2_snr.py)"
    corrected.append(f)
    print(f"  {f}: ({row['dx_mas']:+.0f}, {row['dy_mas']:+.0f}) mas corriges")
print("\nbandes a reprojeter:", " ".join(corrected) if corrected else "aucune")
