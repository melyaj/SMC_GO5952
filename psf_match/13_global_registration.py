"""
13_global_registration.py — global least-squares astrometric closure
=====================================================================
SMC GO-5952 — psf_match module, step 13. Supersedes the incremental
corrections of steps 11/12 (whack-a-mole lesson: per-band corrections
against a single reference leave pair-wise inconsistencies at the
10-25 mas level that the eye catches in blink comparisons).

Method:
  1. candidate stars from F150W; centroids + local peak SNR cached for
     all 14 bands;
  2. for every band pair (i,j): median differential offset d_ij over
     stars with SNR>6 in BOTH bands (differential measurements between
     adjacent-wavelength bands are ~3x more precise than
     each-vs-F150W); pairs kept if N>=10;
  3. weighted least squares for per-band positions p_f minimizing
     sum_ij w_ij |(p_i - p_j) - d_ij|^2, with the NIRCam mean anchored
     to zero (Gaia frame); x and y solved independently;
  4. per-band CRVAL corrections applied (cumulative ASTCORR), bands
     with |correction| < 3 mas left untouched;
  5. after reprojection, rerun with --verify to check all pairwise
     residuals.

Usage: conda activate jwst && python 13_global_registration.py [--verify]
Then:  python 04_reproject_common.py <corrected bands>
"""

import sys
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.wcs import WCS
from scipy import ndimage

from config import ALL_FILTERS, ROOT, TARGET, CONV_DIR, i2d_path

NIRCAM = ['F150W', 'F187N', 'F200W', 'F212N', 'F300M', 'F335M', 'F360M',
          'F444W']
M = ROOT / "products" / "matched"
PIX = 0.11
SNR_MIN, NPAIR_MIN, APPLY_MIN = 6, 10, 3.0
OUT = ROOT / "pipeline" / "psf_match" / "astrometric_global.ecsv"

maps = {f: fits.getdata(M / f"{f}_matched{TARGET}.fits", "SCI")
        for f in ALL_FILTERS}
grid = WCS(fits.getheader(M / f"F150W_matched{TARGET}.fits", "SCI"))
ny, nx = maps['F150W'].shape


def cen(data, x0, y0, box=10):
    c = data[y0 - box:y0 + box + 1, x0 - box:x0 + box + 1].astype(float)
    c = c - np.nanmedian(c)
    c[~np.isfinite(c)] = 0
    c[c < 0] = 0
    yy, xx = np.mgrid[y0 - box:y0 + box + 1, x0 - box:x0 + box + 1]
    s = c.sum()
    return ((xx * c).sum() / s, (yy * c).sum() / s) if s > 0 else (np.nan,) * 2


# ─── 1. star cache ────────────────────────────────────────
_, md, sd = sigma_clipped_stats(maps['F150W'][np.isfinite(maps['F150W'])],
                                sigma=3)
d = np.nan_to_num(maps['F150W'], nan=-1e9)
pk = (d == ndimage.maximum_filter(d, size=25)) & (d > md + 15 * sd)
ys, xs = np.where(pk)
cand = [(x, y) for y, x in zip(ys, xs)
        if 40 < x < nx - 40 and 40 < y < ny - 40]

cache = {}          # (filter, star) -> (cx, cy) or None
snr = {}
for f in ALL_FILTERS:
    for x, y in cand:
        w = maps[f][y - 10:y + 11, x - 10:x + 11]
        if w.shape != (21, 21) or not np.isfinite(w).all():
            snr[f, (x, y)] = -1
            continue
        bg = np.nanmedian(w)
        lsd = 1.4826 * np.nanmedian(np.abs(w - bg))
        snr[f, (x, y)] = (np.nanmax(w[7:14, 7:14]) - bg) / (lsd + 1e-9)
        if snr[f, (x, y)] >= SNR_MIN:
            c = cen(maps[f], x, y)
            cache[f, (x, y)] = c if all(np.isfinite(v) for v in c) and \
                abs(c[0] - x) < 3 and abs(c[1] - y) < 3 else None
print(f"{len(cand)} etoiles candidates, cache calcule")

# ─── 2. pairwise differentials ────────────────────────────
pairs = []
for i, fa in enumerate(ALL_FILTERS):
    for fb in ALL_FILTERS[i + 1:]:
        offs = [(cache[fa, s][0] - cache[fb, s][0],
                 cache[fa, s][1] - cache[fb, s][1])
                for s in cand
                if cache.get((fa, s)) and cache.get((fb, s))]
        if len(offs) < NPAIR_MIN:
            continue
        o = np.array(offs) * PIX * 1e3
        dxy = np.median(o, axis=0)
        err = 1.4826 * np.median(np.abs(o - dxy), axis=0).mean() / \
            np.sqrt(len(o))
        pairs.append((fa, fb, dxy[0], dxy[1], max(err, 1.0), len(o)))
print(f"{len(pairs)} paires mesurees (N>={NPAIR_MIN})")

# ─── 3. weighted least squares ────────────────────────────
idx = {f: i for i, f in enumerate(ALL_FILTERS)}
nf = len(ALL_FILTERS)
sol = {}
for axis, k in (('x', 2), ('y', 3)):
    A, b, w = [], [], []
    for fa, fb, dx, dy, err, n in pairs:
        row = np.zeros(nf)
        row[idx[fa]], row[idx[fb]] = 1, -1
        A.append(row)
        b.append((dx, dy)[k - 2])
        w.append(1 / err)
    # anchor: mean of NIRCam bands = 0
    row = np.zeros(nf)
    for f in NIRCAM:
        row[idx[f]] = 1
    A.append(row)
    b.append(0.0)
    w.append(1000.0)
    A, b, w = np.array(A), np.array(b), np.array(w)
    sol[axis] = np.linalg.lstsq(A * w[:, None], b * w, rcond=None)[0]

print(f"\n{'filtre':8s} {'px(mas)':>8s} {'py(mas)':>8s}  correction?")
rows = []
to_fix = []
for f in ALL_FILTERS:
    px, py = sol['x'][idx[f]], sol['y'][idx[f]]
    fix = bool(np.hypot(px, py) > APPLY_MIN)
    ra_m, dec_m = grid.pixel_to_world_values(nx / 2 + px / PIX / 1e3,
                                             ny / 2 + py / PIX / 1e3)
    ra_t, dec_t = grid.pixel_to_world_values(nx / 2, ny / 2)
    rows.append([f, px, py, fix, float(ra_t - ra_m), float(dec_t - dec_m)])
    if fix:
        to_fix.append(f)
    print(f"{f:8s} {px:+8.1f} {py:+8.1f}  {'OUI' if fix else 'non'}")

tab = Table(rows=rows, names=['filter', 'px_mas', 'py_mas', 'applied',
                              'dCRVAL1_deg', 'dCRVAL2_deg'])
tab.meta['comment'] = [
    'Global least-squares band positions from all pairwise differential',
    'offsets (SNR>6 stars per pair); anchor: NIRCam mean = 0 (Gaia).',
    'Corrections applied when |p| > 3 mas, cumulative with steps 11/12.']
tab.write(OUT, format='ascii.ecsv', overwrite=True)

# residuals of the fit
print("\npaires les moins bien ajustees (residu / erreur):")
res = []
for fa, fb, dx, dy, err, n in pairs:
    rx = dx - (sol['x'][idx[fa]] - sol['x'][idx[fb]])
    ry = dy - (sol['y'][idx[fa]] - sol['y'][idx[fb]])
    res.append((np.hypot(rx, ry) / err, fa, fb, rx, ry, err))
for s, fa, fb, rx, ry, err in sorted(res, reverse=True)[:5]:
    print(f"  {fa}-{fb}: ({rx:+.1f}, {ry:+.1f}) ± {err:.1f} mas [{s:.1f}σ]")

if "--verify" in sys.argv:
    sys.exit(0)

print("\napplication:")
for row in tab:
    if not row['applied']:
        continue
    f = row['filter']
    dra, ddec = float(row['dCRVAL1_deg']), float(row['dCRVAL2_deg'])
    targets = [i2d_path(f)]
    if f != TARGET:
        targets.append(CONV_DIR / f"{f}_conv_{TARGET}.fits")
    for path in targets:
        with fits.open(path, mode='update') as h:
            for ext in ('SCI', 'ERR'):
                if ext in h and 'CRVAL1' in h[ext].header:
                    hdr = h[ext].header
                    hdr['CRVAL1'] += dra
                    hdr['CRVAL2'] += ddec
                    hdr['ASTCORR1'] = (hdr.get('ASTCORR1', 0.0) + dra,
                                       '[deg] cumulative dCRVAL1 (11+12+13)')
                    hdr['ASTCORR2'] = (hdr.get('ASTCORR2', 0.0) + ddec,
                                       '[deg] cumulative dCRVAL2 (11+12+13)')
            h[0].header['HISTORY'] = \
                'global least-squares registration (13_global_registration.py)'
    print(f"  {f}: ({row['px_mas']:+.1f}, {row['py_mas']:+.1f}) mas retires")
print("\nbandes a reprojeter:", " ".join(to_fix) if to_fix else "aucune")
