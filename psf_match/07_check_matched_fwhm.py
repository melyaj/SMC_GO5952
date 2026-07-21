"""
07_check_matched_fwhm.py — end-to-end verification of the PSF matching
======================================================================
Measures the stellar FWHM directly in the analysis_ready maps. If the
matching worked, stars in all 14 filters have the F2100W width (~0.67",
~6.1 px at 0.11"/px), against native widths of 0.05-0.5".

Per filter:
  1. DAOStarFinder on the matched map, keep bright/isolated/unsaturated stars
  2. 2D Gaussian fit on 25x25 px cutouts -> FWHM (arcsec)
  3. sigma-clipped median over the stars

Also overlays the normalized radial profile of one star (common to a red and
a blue filter where possible) across all filters.

Output: figures/matched_fwhm_check.png + printed table.

Usage:
    python 07_check_matched_fwhm.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from astropy.io import fits
from astropy.stats import sigma_clipped_stats, sigma_clip
from photutils.detection import DAOStarFinder

from config import ALL_FILTERS, ANALYSIS_DIR, FIG_DIR, TARGET

PIX = 0.11  # arcsec/px, common grid
HALF = 12   # cutout half-size (px)


def gauss2d(xy, x0, y0, sx, sy, amp, off):
    x, y = xy
    return (off + amp * np.exp(-((x - x0) ** 2 / (2 * sx ** 2)
                                 + (y - y0) ** 2 / (2 * sy ** 2)))).ravel()


def fit_star_fwhm(cut):
    ny, nx = cut.shape
    y, x = np.mgrid[0:ny, 0:nx]
    p0 = (nx / 2, ny / 2, 3.0, 3.0, np.nanmax(cut), np.nanmedian(cut))
    popt, _ = opt.curve_fit(gauss2d, (x, y), np.nan_to_num(cut).ravel(), p0=p0)
    sig = np.sqrt(abs(popt[2]) * abs(popt[3]))  # geometric mean
    return 2.355 * sig * PIX


def measure_filter(filt, nstars=12):
    sci = fits.open(ANALYSIS_DIR / f"{filt}_matched{TARGET}.fits")["SCI"].data
    mean, med, std = sigma_clipped_stats(sci, sigma=3.0, maxiters=3)
    tbl = DAOStarFinder(fwhm=6.0, threshold=30 * std)(np.nan_to_num(sci - med))
    if tbl is None:
        return np.nan, 0, sci
    tbl.sort("flux")
    tbl.reverse()

    ny, nx = sci.shape
    fwhms, used = [], []
    for row in tbl:
        x, y = row["xcentroid"], row["ycentroid"]
        if not (HALF + 5 < x < nx - HALF - 5 and HALF + 5 < y < ny - HALF - 5):
            continue
        if any(np.hypot(x - ux, y - uy) < 30 for ux, uy in used):
            continue
        cut = sci[int(y) - HALF:int(y) + HALF + 1, int(x) - HALF:int(x) + HALF + 1]
        if not np.all(np.isfinite(cut)):
            continue
        try:
            f = fit_star_fwhm(cut)
        except Exception:
            continue
        if 0.2 < f < 2.0:  # reject blends / residual junk
            fwhms.append(f)
            used.append((x, y))
        if len(fwhms) >= nstars:
            break
    clipped = sigma_clip(fwhms, sigma=2.5, maxiters=3)
    return np.mean(clipped.data[~clipped.mask]), np.sum(~clipped.mask), sci


def radial_profile(img, x, y, rmax_px=12):
    ny, nx = img.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.hypot(xx - x, yy - y)
    mask = r < rmax_px
    med = np.nanmedian(img[(r > 2 * rmax_px) & (r < 3 * rmax_px)])
    prof_r = r[mask]
    prof_v = img[mask] - med
    order = np.argsort(prof_r)
    return prof_r[order] * PIX, prof_v[order]


if __name__ == "__main__":
    results = {}
    maps = {}
    for filt in ALL_FILTERS:
        fw, n, sci = measure_filter(filt)
        results[filt] = (fw, n)
        maps[filt] = sci

    ref = results[TARGET][0]
    head_fwhm = 'FWHM (")'
    head_vs = "vs " + TARGET
    print(f"\n{'filtre':8s} {head_fwhm:>9s} {'N*':>4s} {head_vs:>10s}")
    for filt in ALL_FILTERS:
        fw, n = results[filt]
        print(f"{filt:8s} {fw:9.3f} {n:4d} {100 * (fw - ref) / ref:+9.1f} %")

    # radial profile of one bright star, measured in every filter:
    # pick the brightest isolated star of the TARGET map, refit position per map
    sci_t = maps[TARGET]
    mean, med, std = sigma_clipped_stats(sci_t, sigma=3.0, maxiters=3)
    tbl = DAOStarFinder(fwhm=6.0, threshold=50 * std)(np.nan_to_num(sci_t - med))
    tbl.sort("flux")
    tbl.reverse()
    ny, nx = sci_t.shape
    star = next(r for r in tbl
                if 50 < r["xcentroid"] < nx - 50 and 50 < r["ycentroid"] < ny - 50)
    x0, y0 = star["xcentroid"], star["ycentroid"]
    print(f"\nprofil radial sur l'etoile ({x0:.1f}, {y0:.1f})")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    cmap = plt.get_cmap("turbo")
    for i, filt in enumerate(ALL_FILTERS):
        r, v = radial_profile(maps[filt], x0, y0)
        peak = np.nanmax(v[r < 0.3])
        if not np.isfinite(peak) or peak <= 0:
            continue
        ax1.plot(r, v / peak, ".", ms=2, alpha=0.6,
                 color=cmap(i / (len(ALL_FILTERS) - 1)), label=filt)
    ax1.set_xlabel("r (arcsec)")
    ax1.set_ylabel("profil normalise")
    ax1.set_xlim(0, 1.3)
    ax1.set_ylim(-0.1, 1.05)
    ax1.axvline(0.674 / 2, color="k", ls=":", lw=1, label="HWHM F2100W")
    ax1.legend(fontsize=7, ncol=2)
    ax1.set_title(f"meme etoile, 14 filtres matches (etoile {x0:.0f},{y0:.0f})")

    fw = [results[f][0] for f in ALL_FILTERS]
    ax2.bar(range(len(ALL_FILTERS)), fw,
            color=[cmap(i / (len(ALL_FILTERS) - 1)) for i in range(len(ALL_FILTERS))])
    ax2.axhline(ref, color="k", ls="--", lw=1, label=f"{TARGET} = {ref:.3f}\"")
    ax2.axhline(0.674, color="gray", ls=":", lw=1, label='0.674" (JDox)')
    ax2.set_xticks(range(len(ALL_FILTERS)))
    ax2.set_xticklabels(ALL_FILTERS, rotation=90)
    ax2.set_ylabel('FWHM stellaire mesuree (")')
    ax2.legend()
    ax2.set_title("FWHM moyenne des etoiles par filtre (analysis_ready)")
    fig.tight_layout()
    out = FIG_DIR / "matched_fwhm_check.png"
    fig.savefig(out, dpi=140)
    print(f"-> {out}")
