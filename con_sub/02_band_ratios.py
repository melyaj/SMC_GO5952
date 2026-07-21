"""
02_band_ratios.py — PAH band ratio maps
========================================
SMC GO-5952 — con_sub module, step 02.

Pixel-by-pixel PAH band ratio maps from the continuum-subtracted maps
(step 01), following Tarantino+2025 Sect. 3.6 (there computed on clumps;
here full maps, the SMC data are deep enough):

  3.3/11.3  — tracer of the PAH size distribution
  7.7/11.3  — tracer of the PAH ionization state
  3.3/7.7   — mixed size/ionization tracer

Pixels are kept where BOTH bands have SNR > SNR_CUT (this also rejects
the over-subtracted cores of bright compact sources, which go negative).
Ratio uncertainty: err = R * sqrt((eA/A)^2 + (eB/B)^2).

Inputs : products/pah/{F335M,F770W,F1130W}_pah.fits
Outputs: products/pah/ratio_{name}.fits  (RATIO, RATIO_ERR)
         products/pah/band_ratios_summary.png

Usage: conda activate jwst && python 02_band_ratios.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from astropy.io import fits

PAH_DIR = os.path.expanduser('~/SMC_GO5952/products/pah')
SNR_CUT = 3.0

BANDS = {'3.3': 'F335M', '7.7': 'F770W', '11.3': 'F1130W'}
RATIOS = [('3.3', '11.3'), ('7.7', '11.3'), ('3.3', '7.7')]

pah, err, hdr = {}, {}, {}
for band, filt in BANDS.items():
    with fits.open(os.path.join(PAH_DIR, f'{filt}_pah.fits')) as h:
        pah[band] = h['PAH'].data.astype(float)
        err[band] = h['PAH_ERR'].data.astype(float)
        hdr[band] = h['PAH'].header

results = {}
for a, b in RATIOS:
    good = (pah[a] / err[a] > SNR_CUT) & (pah[b] / err[b] > SNR_CUT)
    ratio = np.where(good, pah[a] / pah[b], np.nan)
    rerr = np.abs(ratio) * np.sqrt((err[a] / pah[a]) ** 2 +
                                   (err[b] / pah[b]) ** 2)
    rerr = np.where(good, rerr, np.nan)
    name = f'{a.replace(".", "")}_{b.replace(".", "")}'
    results[(a, b)] = (ratio, rerr)

    h = hdr[a].copy()
    h['RATIO'] = (f'PAH {a}/{b} um', 'band ratio')
    h['SNRCUT'] = (SNR_CUT, 'SNR cut applied to both bands')
    hdus = fits.HDUList([fits.PrimaryHDU()])
    for extname, arr in [('RATIO', ratio), ('RATIO_ERR', rerr)]:
        hh = h.copy(); hh['EXTNAME'] = extname
        hdus.append(fits.ImageHDU(arr.astype(np.float32), header=hh))
    hdus.writeto(os.path.join(PAH_DIR, f'ratio_{name}.fits'), overwrite=True)

    v = ratio[np.isfinite(ratio)]
    print(f'PAH {a}/{b}: {v.size} px kept ({100 * v.size / good.size:.0f}% of map), '
          f'median = {np.median(v):.3f}, 16-84% = [{np.percentile(v, 16):.3f}, '
          f'{np.percentile(v, 84):.3f}], median err = {np.nanmedian(rerr):.3f}')

# ─── Figure: maps + histograms ────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(19, 11),
                         gridspec_kw={'height_ratios': [3, 1]})
for j, (a, b) in enumerate(RATIOS):
    ratio, rerr = results[(a, b)]
    v = ratio[np.isfinite(ratio)]
    vmin, vmax = np.percentile(v, [2, 98])
    ax = axes[0, j]
    m = ax.imshow(ratio, origin='lower', cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
    plt.colorbar(m, ax=ax, fraction=0.046)
    ax.set_title(f'PAH {a} / {b} $\\mu$m   (SNR>{SNR_CUT:.0f} dans les 2 bandes)',
                 fontsize=13)
    ax.set_facecolor('0.85')
    ax.set_xticks([]); ax.set_yticks([])

    axh = axes[1, j]
    axh.hist(v, bins=100, range=(vmin - 0.3 * (vmax - vmin),
                                 vmax + 0.3 * (vmax - vmin)),
             color='#555', histtype='stepfilled', alpha=0.8)
    axh.axvline(np.median(v), color='crimson', ls='--',
                label=f'mediane = {np.median(v):.3f}')
    axh.set_xlabel(f'PAH {a}/{b}')
    axh.set_yticks([])
    axh.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(PAH_DIR, 'band_ratios_summary.png'), dpi=120,
            bbox_inches='tight')
print(f'\nWrote ratio maps and figure to {PAH_DIR}')
