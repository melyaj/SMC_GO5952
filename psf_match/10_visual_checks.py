"""
10_visual_checks.py — visual QA figures for the matched maps
=============================================================
SMC GO-5952 — psf_match module, step 10.

Three figures on products/matched:
  gallery_matched_14.png  — all 14 maps, common frame, asinh
  star_zoom_14bands.png   — validation star (1046,508): same pixel,
                            same 3.5" frame in all bands; cyan circle
                            = target FWHM 0.674"; crosshair = identical
                            position (PSF homogeneity + alignment)
  alignment_visual.png    — cross-instrument contour overlays + RGB
                            (misalignment would show as color fringes)

Usage: conda activate jwst && python 10_visual_checks.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.visualization import simple_norm, make_lupton_rgb

from config import ALL_FILTERS, ROOT, TARGET, FIG_DIR

M = ROOT / "products" / "matched"
maps = {f: fits.getdata(M / f"{f}_matched{TARGET}.fits", "SCI")
        for f in ALL_FILTERS}
PIX = 0.11
STAR = (1046, 508)

# ── 1. gallery ────────────────────────────────────────────
fig, axes = plt.subplots(2, 7, figsize=(26, 8.5))
for ax, f in zip(axes.ravel(), ALL_FILTERS):
    d = maps[f]
    ax.imshow(d, origin='lower', cmap='afmhot',
              norm=simple_norm(d, 'asinh', percent=99.5))
    ax.set_title(f, fontsize=13)
    ax.set_xticks([]); ax.set_yticks([])
plt.suptitle('Les 14 cartes matchées (grille commune 0.11"/px north-up, '
             'PSF F2100W)', fontsize=14)
plt.tight_layout()
plt.savefig(FIG_DIR / 'gallery_matched_14.png', dpi=95, bbox_inches='tight')
plt.close()

# ── 2. star zooms ─────────────────────────────────────────
X0, Y0, HW = *STAR, 16
fig, axes = plt.subplots(2, 7, figsize=(24, 7.5))
for ax, f in zip(axes.ravel(), ALL_FILTERS):
    cut = maps[f][Y0 - HW:Y0 + HW + 1, X0 - HW:X0 + HW + 1]
    ax.imshow(cut, origin='lower', cmap='magma',
              norm=simple_norm(cut, 'asinh', percent=99.9),
              extent=[-HW * PIX, HW * PIX, -HW * PIX, HW * PIX])
    ax.add_patch(plt.Circle((0, 0), 0.674 / 2, fill=False, color='cyan',
                            lw=1.2, ls='--'))
    ax.axhline(0, color='w', lw=0.4, alpha=0.6)
    ax.axvline(0, color='w', lw=0.4, alpha=0.6)
    ax.set_title(f, fontsize=12)
    ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
    ax.tick_params(labelsize=7)
fig.suptitle(f'Etoile {STAR} : meme pixel, meme cadre 3.5" dans les 14 '
             'bandes — cercle cyan = FWHM cible 0.674", reticule = '
             'position identique', fontsize=13)
plt.tight_layout()
plt.savefig(FIG_DIR / 'star_zoom_14bands.png', dpi=110, bbox_inches='tight')
plt.close()

# ── 3. alignment overlays ─────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
zx, zy, hw = *STAR, 40
ax = axes[0]
base = maps['F2100W'][zy - hw:zy + hw, zx - hw:zx + hw]
ax.imshow(base, origin='lower', cmap='gray',
          norm=simple_norm(base, 'asinh', percent=99.8))
for f, c in [('F150W', 'cyan'), ('F770W', 'orange')]:
    cut = maps[f][zy - hw:zy + hw, zx - hw:zx + hw]
    ax.contour(cut, levels=np.nanpercentile(cut, [97, 99, 99.8]),
               colors=c, linewidths=1.0)
ax.set_title("fond: F2100W | contours: F150W (cyan), F770W (orange)\n"
             "zoom 8.8\" sur l'etoile")
ax.set_xticks([]); ax.set_yticks([])

zx, zy, hw = 620, 850, 90
ax = axes[1]
base = maps['F770W'][zy - hw:zy + hw, zx - hw:zx + hw]
ax.imshow(base, origin='lower', cmap='gray',
          norm=simple_norm(base, 'asinh', percent=99.5))
for f, c in [('F200W', 'cyan'), ('F1500W', 'red')]:
    cut = maps[f][zy - hw:zy + hw, zx - hw:zx + hw]
    ax.contour(cut, levels=np.nanpercentile(cut, [98, 99.5]),
               colors=c, linewidths=0.9)
ax.set_title('fond: F770W | contours: F200W (cyan), F1500W (rouge)\n'
             'region 19.8"')
ax.set_xticks([]); ax.set_yticks([])

zx, zy, hw = *STAR, 60


def ch(f):
    c = maps[f][zy - hw:zy + hw, zx - hw:zx + hw].copy()
    c = np.nan_to_num(c - np.nanmedian(c))
    return np.clip(c / np.nanpercentile(c, 99.7), 0, 1)


rgb = make_lupton_rgb(ch('F2100W'), ch('F770W'), ch('F200W'),
                      stretch=0.7, Q=8)
axes[2].imshow(rgb, origin='lower')
axes[2].set_title('RGB: R=F2100W, G=F770W, B=F200W (13.2")\n'
                  'un desalignement ferait des franges colorees')
axes[2].set_xticks([]); axes[2].set_yticks([])
plt.tight_layout()
plt.savefig(FIG_DIR / 'alignment_visual.png', dpi=115, bbox_inches='tight')
plt.close()
print('3 figures ->', FIG_DIR)
