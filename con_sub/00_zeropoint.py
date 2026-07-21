"""
00_zeropoint.py — Zero-point homogenization of the PSF-matched maps
====================================================================
SMC GO-5952 — con_sub module, step 00.

Problem: each filter carries its own arbitrary background zero
(NIRCam: zodiacal pedestal never subtracted; MIRI: per-filter 2D
polynomial sky subtraction with filter-dependent masks). Additive
offsets between bands propagate directly into the k-method continuum
subtraction and band ratios.

Fix: define ONE common dark reference region on the matched maps
(largest connected component of the darkest 2% of the smoothed
F770W+F2100W dust tracer, within the common valid footprint), measure
the sigma-clipped median of each band there, and subtract it as a
per-filter constant. Convention: flux = 0 in the dark cavity, same for
all 14 bands (equivalent to Tarantino+2025 off-galaxy reference).

The zone-to-zone scatter (next 3 largest dark components) is stored as
the systematic uncertainty of the zero point (ZPSYS) — it is NOT added
to the ERR maps; carry it separately in the PAH error budget.

Inputs : analysis_ready/{filt}_matchedF2100W.fits  (SCI, ERR)
Outputs: analysis_ready/zp/{filt}_matchedF2100W_zp.fits
         analysis_ready/zp/zeropoint_offsets.ecsv
         analysis_ready/zp/zeropoint_refzones.fits   (zone label map)
         analysis_ready/zp/zeropoint_dark_zones.png

Usage: conda activate jwst && python 00_zeropoint.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.visualization import simple_norm
from scipy import ndimage

# ─── Configuration ────────────────────────────────────────
FILTERS = ['F150W', 'F187N', 'F200W', 'F212N', 'F300M', 'F335M', 'F360M',
           'F444W', 'F560W', 'F770W', 'F1000W', 'F1130W', 'F1500W', 'F2100W']
IN_DIR      = os.path.expanduser('~/SMC_GO5952/analysis_ready')
OUT_DIR     = os.path.join(IN_DIR, 'zp')
DARK_PCT    = 2      # darkest percentile of F770W+F2100W defining candidate dark pixels
SMOOTH_PX   = 5      # boxcar smoothing before thresholding (avoid noise selection)
N_ZONES     = 4      # zone 1 = reference; zones 2..N -> systematic scatter
CLIP_SIGMA  = 3

os.makedirs(OUT_DIR, exist_ok=True)


def in_file(filt):
    return os.path.join(IN_DIR, f'{filt}_matchedF2100W.fits')


def out_file(filt):
    return os.path.join(OUT_DIR, f'{filt}_matchedF2100W_zp.fits')


# ─── 1. Build the dark reference zones ────────────────────
print('Loading SCI maps...')
sci = {f: fits.getdata(in_file(f), 'SCI') for f in FILTERS}

valid = np.all([np.isfinite(sci[f]) for f in FILTERS], axis=0)
print(f'  common valid footprint: {valid.sum()} px ({100 * valid.mean():.0f}%)')

comb = ndimage.uniform_filter(
    np.nan_to_num(sci['F770W'] + sci['F2100W'], nan=1e9), SMOOTH_PX)
thr = np.percentile(comb[valid], DARK_PCT)
dark = valid & (comb < thr)

lab, nlab = ndimage.label(dark)
sizes = ndimage.sum(dark, lab, range(1, nlab + 1))
zones = np.argsort(sizes)[::-1][:N_ZONES] + 1   # labels, largest first
print(f'  dark mask: {dark.sum()} px (threshold {thr:.3f} MJy/sr), '
      f'{nlab} components')
for i, l in enumerate(zones):
    yy, xx = np.where(lab == l)
    print(f'    zone {i + 1}: {int(sizes[l - 1])} px, '
          f'center ~({np.median(xx):.0f},{np.median(yy):.0f})')

ref = lab == zones[0]   # zone 1 = the reference

# ─── 2. Measure per-filter offsets ────────────────────────
rows = []
for f in FILTERS:
    _, med_ref, _ = sigma_clipped_stats(sci[f][ref], sigma=CLIP_SIGMA)
    zone_meds = []
    for l in zones:
        _, zm, _ = sigma_clipped_stats(sci[f][lab == l], sigma=CLIP_SIGMA)
        zone_meds.append(zm)
    sys_unc = np.std(zone_meds)
    rows.append([f, med_ref] + zone_meds + [sys_unc])
    print(f'{f:8s} offset = {med_ref:+8.4f}  sys = {sys_unc:.4f} MJy/sr')

colnames = (['filter', 'offset'] +
            [f'zone{i + 1}_median' for i in range(N_ZONES)] + ['sys_scatter'])
table = Table(rows=rows, names=colnames)
table.meta['comment'] = [
    'Zero-point offsets measured in the common dark reference region',
    f'(zone 1 = largest component of darkest {DARK_PCT}% of smoothed',
    f'F770W+F2100W, sigma-clipped median at {CLIP_SIGMA} sigma).',
    'Science convention: SCI_zp = SCI - offset (flux = 0 in zone 1).',
    'sys_scatter = std of the zone medians = systematic zero uncertainty.',
    'Units: MJy/sr.',
]
table.write(os.path.join(OUT_DIR, 'zeropoint_offsets.ecsv'),
            format='ascii.ecsv', overwrite=True)

# ─── 3. Write the zp FITS files ───────────────────────────
for row in table:
    f, off, sysu = row['filter'], row['offset'], row['sys_scatter']
    with fits.open(in_file(f)) as hdu:
        hdu['SCI'].data = hdu['SCI'].data - off
        for ext in ('SCI', 'ERR'):
            hdu[ext].header['ZPOFF'] = (
                off, '[MJy/sr] zero-point offset subtracted from SCI')
            hdu[ext].header['ZPSYS'] = (
                sysu, '[MJy/sr] systematic zero uncertainty (zone scatter)')
            hdu[ext].header['ZPNPIX'] = (
                int(ref.sum()), 'pixels in dark reference zone 1')
        hdu[0].header['HISTORY'] = (
            'Zero-point homogenized (con_sub/00_zeropoint.py): '
            'SCI - ZPOFF, common dark-cavity reference')
        hdu.writeto(out_file(f), overwrite=True)
print(f'\nWrote {len(FILTERS)} zp files to {OUT_DIR}')

# ─── 4. Save the zone map ─────────────────────────────────
zone_map = np.zeros(lab.shape, dtype=np.int16)
for i, l in enumerate(zones):
    zone_map[lab == l] = i + 1
hdr = fits.getheader(in_file('F770W'), 'SCI')
fits.writeto(os.path.join(OUT_DIR, 'zeropoint_refzones.fits'),
             zone_map, header=hdr, overwrite=True)

# ─── 5. Figure ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 9))
norm = simple_norm(sci['F770W'], 'asinh', percent=99.5)
ax.imshow(sci['F770W'], origin='lower', cmap='afmhot', norm=norm)
colors = ['cyan', 'lime', 'deepskyblue', 'magenta']
for i, l in enumerate(zones):
    ax.contour(lab == l, levels=[0.5], colors=colors[i], linewidths=1.5)
    yy, xx = np.where(lab == l)
    ax.annotate(f'zone {i + 1}', (np.median(xx), np.max(yy) + 15),
                color=colors[i], fontsize=12, ha='center')
ax.set_title(f'F770W (asinh) — dark reference zones '
             f'(darkest {DARK_PCT}% of F770W+F2100W)')
plt.savefig(os.path.join(OUT_DIR, 'zeropoint_dark_zones.png'),
            dpi=150, bbox_inches='tight')

# ─── 6. Verification ──────────────────────────────────────
print('\nVerification (zone 1 median after correction, should be ~0):')
for f in FILTERS:
    d = fits.getdata(out_file(f), 'SCI')
    _, med, _ = sigma_clipped_stats(d[ref], sigma=CLIP_SIGMA)
    print(f'  {f:8s} {med:+.2e} MJy/sr')
