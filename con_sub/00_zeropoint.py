"""
00_zeropoint.py — zero-point VERIFICATION on the matched maps
==============================================================
SMC GO-5952 — con_sub module, step 00.

ROLE CHANGE (2026-07-22): the zero-point constants are now applied at
mosaic level BEFORE PSF matching (psf_match/00_subtract_zeropoint.py,
offsets in psf_match/zeropoint_offsets.ecsv). This script no longer
produces files; it VERIFIES that the final matched maps satisfy the
zero-point convention: sigma-clipped median ~ 0 in the dark reference
cavity (zone 1) for all 14 bands.

The reference-zone map (zeropoint_refzones.fits) is regenerated here
from the matched maps if absent (same definition as the original
measurement: largest connected component of the darkest 2% of the
5-px-smoothed F770W+F2100W within the common footprint).

Usage: conda activate jwst && python 00_zeropoint.py
"""

import os
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from scipy import ndimage

FILTERS = ['F150W', 'F187N', 'F200W', 'F212N', 'F300M', 'F335M', 'F360M',
           'F444W', 'F560W', 'F770W', 'F1000W', 'F1130W', 'F1500W', 'F2100W']
MATCHED = os.path.expanduser('~/SMC_GO5952/products/matched')
REFZONES = os.path.join(MATCHED, 'zeropoint_refzones.fits')
TOL = 0.02   # MJy/sr — tolerance on the dark-cavity median


def m(f):
    return fits.getdata(os.path.join(MATCHED, f'{f}_matchedF2100W.fits'),
                        'SCI').astype(float)


if not os.path.exists(REFZONES):
    print('building reference-zone map...')
    sci = {f: m(f) for f in FILTERS}
    valid = np.all([np.isfinite(sci[f]) for f in FILTERS], axis=0)
    comb = ndimage.uniform_filter(
        np.nan_to_num(sci['F770W'] + sci['F2100W'], nan=1e9), 5)
    dark = valid & (comb < np.percentile(comb[valid], 2))
    lab, nlab = ndimage.label(dark)
    sizes = ndimage.sum(dark, lab, range(1, nlab + 1))
    zones = np.argsort(sizes)[::-1][:4] + 1
    zone_map = np.zeros(lab.shape, dtype=np.int16)
    for i, l in enumerate(zones):
        zone_map[lab == l] = i + 1
    hdr = fits.getheader(os.path.join(
        MATCHED, 'F770W_matchedF2100W.fits'), 'SCI')
    fits.writeto(REFZONES, zone_map, header=hdr, overwrite=True)

ref = fits.getdata(REFZONES) == 1
print(f'zone 1: {ref.sum()} px\n')
print('filtre   mediane_cavite  ZPOFF_header  statut')
ok = True
for f in FILTERS:
    path = os.path.join(MATCHED, f'{f}_matchedF2100W.fits')
    d = fits.getdata(path, 'SCI')
    zpoff = fits.getheader(path, 'SCI').get('ZPOFF', np.nan)
    _, med, _ = sigma_clipped_stats(d[ref], sigma=3)
    status = 'OK' if abs(med) < TOL else 'HORS TOLERANCE'
    ok &= abs(med) < TOL
    print(f'{f:8s} {med:+12.4f}   {zpoff:+10.4f}   {status}')
print('\nVERIFICATION ' + ('PASSEE' if ok else 'ECHOUEE') +
      f' (tolerance {TOL} MJy/sr)')
