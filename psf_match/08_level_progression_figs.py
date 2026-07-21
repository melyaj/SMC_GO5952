"""
08_level_progression_figs.py — visual QA of the 5 processing levels
====================================================================
SMC GO-5952 — psf_match module, step 08 (quality assurance).

For a given filter, builds one figure showing the mosaic at each
processing level (levels/01..05) and, below, WHAT each transition
changed, with printed QA metrics:

  01 stage3      -> 02 skysub     : removed sky model (MIRI only)
  02 skysub      -> 03 zeropoint  : constant pedestal (must equal ZPOFF)
  03 zeropoint   -> 04 convolved  : PSF broadening (star cutout + flux
                                    conservation in r=3" aperture)
  04 convolved   -> 05 matched    : reprojection to the common north-up
                                    grid (aperture flux conserved)

Usage: conda activate jwst && python 08_level_progression_figs.py F770W F335M
       (no args: F770W F335M)
Output: psf_matching/figures/levels_{FILT}.png
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.visualization import simple_norm
from astropy.wcs import WCS

from config import ROOT, MIRI_FILTERS, FIG_DIR, TARGET, i2d_path_raw, i2d_path

MATCHED_DIR = ROOT / "products" / "matched"
CONV_DIR = ROOT / "psf_matching" / "convolved"
STAR_MATCHED_XY = (1046, 508)      # validation star on the common grid


def sky_aperture_sum(data, wcs, ra, dec, r_arcsec=3.0):
    """Sum in a circular aperture defined on the sky (NaN-safe)."""
    x0, y0 = wcs.world_to_pixel_values(ra, dec)
    scale = np.sqrt(np.abs(np.linalg.det(wcs.pixel_scale_matrix))) * 3600
    r_pix = r_arcsec / scale
    yy, xx = np.ogrid[:data.shape[0], :data.shape[1]]
    m = (xx - x0) ** 2 + (yy - y0) ** 2 <= r_pix ** 2
    vals = data[m]
    return np.nansum(vals) * scale ** 2, (float(x0), float(y0), r_pix)


def moment_fwhm(cut):
    """Crude moment-based FWHM (px) of a background-subtracted cutout."""
    c = np.nan_to_num(cut - np.nanmedian(cut))
    c[c < 0] = 0
    tot = c.sum()
    if tot <= 0:
        return np.nan
    yy, xx = np.mgrid[:c.shape[0], :c.shape[1]]
    xb, yb = (xx * c).sum() / tot, (yy * c).sum() / tot
    var = ((xx - xb) ** 2 * c + (yy - yb) ** 2 * c).sum() / (2 * tot)
    return 2.355 * np.sqrt(var)


def qa_figure(filt):
    is_miri = filt in MIRI_FILTERS
    # ---- load the levels ----
    lv = {}
    with fits.open(i2d_path_raw(filt).with_name(
            i2d_path_raw(filt).name)) as h:      # 01 raw stage3
        pass
    raw_path = (ROOT / "data" / ("miri" if is_miri else "nircam") / filt /
                "stage3" / (("miri_" if is_miri else "nircam_") +
                            f"{filt}_final_i2d.fits"))
    lv[1] = fits.open(raw_path)
    if is_miri:
        lv[2] = fits.open(str(raw_path).replace(".fits", "_skysub.fits"))
    lv[3] = fits.open(i2d_path(filt))
    lv[4] = fits.open(CONV_DIR / f"{filt}_conv_{TARGET}.fits")
    lv[5] = fits.open(MATCHED_DIR / f"{filt}_matched{TARGET}.fits")

    d = {k: v["SCI"].data.astype(float) for k, v in lv.items()}
    w = {k: WCS(v["SCI"].header) for k, v in lv.items()}
    zpoff = lv[3]["SCI"].header["ZPOFF"]

    # star sky position from the matched grid
    ra, dec = w[5].pixel_to_world_values(*STAR_MATCHED_XY)

    # ---- QA metrics ----
    qa = []
    _, med1, _ = sigma_clipped_stats(d[1][np.isfinite(d[1])], sigma=3)
    if is_miri:
        _, med2, _ = sigma_clipped_stats(d[2][np.isfinite(d[2])], sigma=3)
        model = lv[2]["SKYMODEL"].data
        qa.append(f"01->02  bg median {med1:+.3f} -> {med2:+.3f} MJy/sr "
                  f"(sky model {np.nanmin(model):+.3f}..{np.nanmax(model):+.3f})")
        base = 2
    else:
        qa.append("01->02  (no NIRCam post-mosaic skysub: level 02 = 01)")
        base = 1
    diff = d[base] - d[3]
    fin = np.isfinite(diff)
    qa.append(f"{'02' if is_miri else '01'}->03  pedestal: measured "
              f"{np.nanmedian(diff[fin]):+.4f} vs ZPOFF {zpoff:+.4f} "
              f"(max dev {np.nanmax(np.abs(diff[fin] - zpoff)):.1e}) MJy/sr")

    f3, ap3 = sky_aperture_sum(d[3], w[3], ra, dec)
    f4, ap4 = sky_aperture_sum(d[4], w[4], ra, dec)
    f5, ap5 = sky_aperture_sum(d[5], w[5], ra, dec)
    scale3 = np.sqrt(np.abs(np.linalg.det(w[3].pixel_scale_matrix))) * 3600
    scale5 = np.sqrt(np.abs(np.linalg.det(w[5].pixel_scale_matrix))) * 3600
    cut = 25
    x3, y3 = int(ap3[0]), int(ap3[1])
    x5, y5 = int(ap5[0]), int(ap5[1])
    star3 = d[3][y3 - cut:y3 + cut, x3 - cut:x3 + cut]
    star4 = d[4][y3 - cut:y3 + cut, x3 - cut:x3 + cut]
    star5 = d[5][y5 - cut:y5 + cut, x5 - cut:x5 + cut]
    qa.append(f"03->04  star FWHM {moment_fwhm(star3) * scale3:.2f}\" -> "
              f"{moment_fwhm(star4) * scale3:.2f}\" (target ~0.67\"); "
              f"aperture flux ratio {f4 / f3:.4f}")
    qa.append(f"04->05  reprojection {d[4].shape}@{scale3:.3f}\" -> "
              f"{d[5].shape}@{scale5:.3f}\" north-up; "
              f"aperture flux ratio {f5 / f4:.4f}")

    # ---- figure ----
    fig, axes = plt.subplots(2, 5, figsize=(24, 10))
    titles = {1: "01 stage3", 2: "02 skysub", 3: "03 zeropoint",
              4: f"04 convolved ({TARGET} PSF)", 5: "05 matched (common grid)"}
    vmin, vmax = -0.5, np.nanpercentile(d[3], 99.0)
    for j, k in enumerate([1, 2, 3, 4, 5]):
        ax = axes[0, j]
        if k == 2 and not is_miri:
            ax.text(0.5, 0.5, "= level 01\n(no NIRCam skysub)",
                    ha='center', va='center', fontsize=13)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(titles[2])
            continue
        ax.imshow(d[k], origin='lower', cmap='afmhot', vmin=vmin, vmax=vmax)
        ax.set_title(titles[k] + f"\nmedian {np.nanmedian(d[k]):+.3f} MJy/sr")
        ax.set_xticks([]); ax.set_yticks([])

    # bottom row: what changed
    ax = axes[1, 0]
    if is_miri:
        m = ax.imshow(lv[2]["SKYMODEL"].data, origin='lower', cmap='viridis')
        plt.colorbar(m, ax=ax, fraction=0.046)
        ax.set_title("01->02 : sky model removed")
    else:
        ax.text(0.5, 0.5, "no skysub step", ha='center', va='center')
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[1, 1]
    m = ax.imshow(diff, origin='lower', cmap='RdBu_r',
                  vmin=zpoff - 0.02, vmax=zpoff + 0.02)
    plt.colorbar(m, ax=ax, fraction=0.046)
    ax.set_title(f"->03 : difference map (flat = {zpoff:+.3f})")
    ax.set_xticks([]); ax.set_yticks([])

    for j, (st, lab) in enumerate([(star3, "star @03 (native PSF)"),
                                   (star4, "star @04 (F2100W PSF)")]):
        ax = axes[1, 2 + j]
        norm = simple_norm(st, 'asinh', percent=99.9)
        ax.imshow(st, origin='lower', cmap='magma', norm=norm)
        ax.set_title(lab)
        ax.set_xticks([]); ax.set_yticks([])

    ax = axes[1, 4]
    norm = simple_norm(star5, 'asinh', percent=99.9)
    ax.imshow(star5, origin='lower', cmap='magma', norm=norm)
    ax.set_title("star @05 (common grid)")
    ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"{filt} — chain progression QA\n" + "  |  ".join(qa),
                 fontsize=11)
    out = FIG_DIR / f"levels_{filt}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"\n{filt}:")
    for line in qa:
        print("  " + line)
    print(f"  -> {out}")
    for v in lv.values():
        v.close()


if __name__ == "__main__":
    for filt in (sys.argv[1:] or ["F770W", "F335M"]):
        qa_figure(filt)
