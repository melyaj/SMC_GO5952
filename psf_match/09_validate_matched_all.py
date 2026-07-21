"""
09_validate_matched_all.py — global validation: PSF matching + alignment
=========================================================================
SMC GO-5952 — psf_match module, step 09 (final QA before band ratios).

Three tests, in decreasing order of purity:

  A. MODEL SPACE (definitive for the PSF): PSF_filt (rotated STPSF
     model) convolved with its Aniano kernel must equal the F2100W
     PSF model. No field systematics. Criterion: |dFWHM| < 5%.
  B. EMPIRICAL (sanity): radial profile of the bright validation star
     (1046, 508) on the matched maps. Carries field systematics
     (nebular background, star color): expect <~10% scatter.
  C. ALIGNMENT: centroid offsets of field stars vs F2100W, in mas.
     Criterion: |median offset| well below the 670 mas beam.

CAUTION (learned the hard way): do NOT select PSF stars by requiring
brightness at both 2 and 21 um — in this field those are embedded
YSOs, intrinsically extended, mimicking unmatched PSFs. Photospheres
are blue; only rare bright stars are usable empirically at 21 um.

Output: psf_matching/figures/validate_matched_all.png + printed tables
Usage: conda activate jwst && python 09_validate_matched_all.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from scipy import ndimage
from scipy.signal import fftconvolve

from config import ALL_FILTERS, ROOT, TARGET, KER_DIR, PSF_DIR, FIG_DIR

MATCHED = ROOT / "products" / "matched"
PIX = 0.11
STAR = (1046, 508)

maps = {f: fits.getdata(MATCHED / f"{f}_matched{TARGET}.fits", "SCI")
        for f in ALL_FILTERS}
ny, nx = maps[TARGET].shape


def radial_fwhm(cut, pix, recenter=True):
    n = cut.shape[0] // 2
    yy, xx = np.mgrid[-n:n + 1, -n:n + 1]
    cx = cy = 0.0
    if recenter:
        core = np.where(np.hypot(xx, yy) < 6,
                        np.clip(cut - np.nanmedian(cut), 0, None), 0)
        if core.sum() > 0:
            cx = (xx * core).sum() / core.sum()
            cy = (yy * core).sum() / core.sum()
    r = np.hypot(xx - cx, yy - cy)
    bg = np.nanmedian(cut[(r > n - 4) & (r < n)]) if recenter else 0.0
    c = cut - bg
    rb = np.arange(0, n - 2)
    p = np.array([np.nanmedian(c[(r >= ri) & (r < ri + 1)]) for ri in rb])
    p = p / p[0]
    below = np.where(p < 0.5)[0]
    if len(below) == 0 or below[0] == 0:
        return np.nan, p
    i = below[0]
    return 2 * (i - 1 + (0.5 - p[i - 1]) / (p[i] - p[i - 1])) * pix, p


# ─── A. model space ───────────────────────────────────────
print("A) MODEL SPACE: PSF_filt (x) kernel vs PSF_F2100W")
t = fits.open(PSF_DIR / f"{TARGET}_stpsf_rot.fits")
tpix = t[0].header["PIXELSCL"]
c0 = t[0].data.shape[0] // 2
fw_t, _ = radial_fwhm(t[0].data[c0 - 90:c0 + 91, c0 - 90:c0 + 91], tpix,
                      recenter=False)
model_dev = {}
for f in ALL_FILTERS:
    if f == TARGET:
        model_dev[f] = 0.0
        continue
    p = fits.open(PSF_DIR / f"{f}_stpsf_rot.fits")
    k = fits.open(KER_DIR / f"{f}_to_{TARGET}_kernel.fits")
    conv = fftconvolve(p[0].data, k[0].data, mode='same')
    n0 = conv.shape[0] // 2
    m = min(90, n0 - 2)
    fw, _ = radial_fwhm(conv[n0 - m:n0 + m + 1, n0 - m:n0 + m + 1],
                        p[0].header["PIXELSCL"], recenter=False)
    model_dev[f] = 100 * (fw - fw_t) / fw_t
ok_a = all(abs(v) < 5 for v in model_dev.values())
for f in ALL_FILTERS:
    print(f"  {f:8s} {model_dev[f]:+6.1f}%")
print(f"  -> {'PASSED' if ok_a else 'FAILED'} (criterion |dFWHM| < 5%)")

# ─── B. empirical bright star ─────────────────────────────
print(f"\nB) EMPIRICAL: star {STAR} on the matched maps")
star_fw = {}
for f in ALL_FILTERS:
    cut = maps[f][STAR[1] - 18:STAR[1] + 19,
                  STAR[0] - 18:STAR[0] + 19].astype(float)
    star_fw[f], _ = radial_fwhm(cut, PIX)
for f in ALL_FILTERS:
    print(f"  {f:8s} {star_fw[f]:.3f}\"")
spread = 100 * (max(star_fw.values()) - min(star_fw.values())) / \
    np.mean(list(star_fw.values()))
print(f"  -> spread {spread:.0f}% (field systematics included; "
      "informative, not a pass/fail)")

# ─── C. alignment ─────────────────────────────────────────
print("\nC) ALIGNMENT: field-star centroids vs F2100W")
det = maps["F200W"]
valid_all = np.all([np.isfinite(maps[f]) for f in ALL_FILTERS], axis=0)
_, med, std = sigma_clipped_stats(det[np.isfinite(det)], sigma=3)
d = np.nan_to_num(det, nan=-1e9)
peak = (d == ndimage.maximum_filter(d, size=25)) & (d > med + 15 * std)
ys, xs = np.where(peak)
cand = []
for y, x in zip(ys, xs):
    if 40 < x < nx - 40 and 40 < y < ny - 40 and \
            valid_all[y - 10:y + 11, x - 10:x + 11].all():
        cand.append((det[y, x], x, y))
cand.sort(reverse=True)


def centroid(data, x0, y0, box=10):
    cut = data[y0 - box:y0 + box + 1, x0 - box:x0 + box + 1].astype(float)
    cut = cut - np.nanmedian(cut)
    cut[~np.isfinite(cut)] = 0
    cut[cut < 0] = 0
    yy, xx = np.mgrid[y0 - box:y0 + box + 1, x0 - box:x0 + box + 1]
    s = cut.sum()
    return ((xx * cut).sum() / s, (yy * cut).sum() / s) if s > 0 else (np.nan,) * 2


cents = {f: [] for f in ALL_FILTERS}
for _, x, y in cand[:30]:
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
ref = np.array(cents[TARGET])
print(f"  {len(ref)} stars; median offsets vs {TARGET} (mas):")
ok_c = True
align = {}
for f in ALL_FILTERS:
    off = (np.array(cents[f]) - ref) * PIX * 1000
    mx, my = np.median(off[:, 0]), np.median(off[:, 1])
    align[f] = (mx, my, off)
    tot = np.hypot(mx, my)
    ok_c &= tot < 100
    print(f"  {f:8s} dx {mx:+6.1f}  dy {my:+6.1f}  |d| {tot:5.1f}"
          f"  ({100 * tot / 670:.0f}% of beam)")
print(f"  -> {'PASSED' if ok_c else 'FAILED'} (criterion |median| < 100 mas "
      "= 15% of the 670 mas beam)")

# ─── figure ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
axes[0].axhspan(-5, 5, color='k', alpha=0.08)
axes[0].plot(range(14), [model_dev[f] for f in ALL_FILTERS], 'o-',
             color='#1a7a4a')
axes[0].set_xticks(range(14))
axes[0].set_xticklabels(ALL_FILTERS, rotation=60, fontsize=8)
axes[0].set_ylabel('dFWHM vs F2100W model (%)')
axes[0].set_title('A. Model space: PSF (x) kernel vs target (±5%)')

axes[1].axhline(star_fw[TARGET], color='k', ls='--', lw=1)
axes[1].plot(range(14), [star_fw[f] for f in ALL_FILTERS], 'o-',
             color='#c33')
axes[1].set_xticks(range(14))
axes[1].set_xticklabels(ALL_FILTERS, rotation=60, fontsize=8)
axes[1].set_ylabel('FWHM (arcsec)')
axes[1].set_title(f'B. Star {STAR} (field systematics included)')

cmap = plt.get_cmap('turbo')
for i, f in enumerate(ALL_FILTERS):
    off = align[f][2]
    axes[2].scatter(off[:, 0], off[:, 1], s=16, color=cmap(i / 13),
                    label=f, alpha=0.8)
axes[2].add_patch(plt.Circle((0, 0), 100, fill=False, ls=':', color='k'))
axes[2].axhline(0, color='k', lw=0.5)
axes[2].axvline(0, color='k', lw=0.5)
axes[2].set_xlabel('Δx (mas)')
axes[2].set_ylabel('Δy (mas)')
axes[2].set_title('C. Centroid offsets (circle = 100 mas = 15% beam)')
axes[2].set_aspect('equal')
axes[2].legend(fontsize=6, ncol=2)
plt.tight_layout()
out = FIG_DIR / "validate_matched_all.png"
plt.savefig(out, dpi=120, bbox_inches='tight')
print(f"\n-> {out}")
