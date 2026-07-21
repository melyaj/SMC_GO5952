"""
04_consub_alternatives.py — alternative continuum subtractions, compared
=========================================================================
SMC GO-5952 — con_sub module, step 04.

Recomputes the three PAH maps with alternative continuum-subtraction
prescriptions and compares them to the fiducial k-method (step 01,
k1 values of Tarantino+2025):

  lin    : plain linear interpolation between the flanking filters
           (no PAH-contamination correction, k -> infinity limit)
  k2     : k-method with the alternative k2 values of Tarantino+2025
           Table 2 (3.3: 4.45+/-0.39, 7.7: 5.84+/-0.73, 11.3:
           10.17+/-1.24)
  sand16 : F335M only — k = 1.6 (slope of Sandstrom+2023 empirical
           F335M PAH color method, quoted by Tarantino as comparable
           to her PAHFIT k)  [placeholder until the exact prescription
           from the literature agents is integrated]

Outputs: analysis_ready/pah/alternatives/{filt}_pah_{method}.fits
         analysis_ready/pah/alternatives/consub_method_comparison.png
         analysis_ready/pah/alternatives/consub_method_stats.ecsv

Usage: conda activate jwst && python 04_consub_alternatives.py
"""

import os
import types
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.table import Table

# reuse the (corrected) production functions from step 01
SRC = os.path.expanduser('~/SMC_GO5952/pipeline/con_sub/01_consub_kmethod.py')
src = open(SRC).read()
mod = types.ModuleType('kfuncs')
mod.np = np
exec(src[src.index('def get_pah_low'):src.index('# ─── I/O helpers')],
     mod.__dict__)
get_pah_low, get_pah_up = mod.get_pah_low, mod.get_pah_up

ZP_DIR  = os.path.expanduser('~/SMC_GO5952/analysis_ready/zp')
PAH_DIR = os.path.expanduser('~/SMC_GO5952/analysis_ready/pah')
OUT_DIR = os.path.join(PAH_DIR, 'alternatives')
os.makedirs(OUT_DIR, exist_ok=True)

PIVOT = {'F300M': 2.996, 'F335M': 3.365, 'F360M': 3.621,
         'F560W': 5.635, 'F770W': 7.639, 'F1000W': 9.953,
         'F1130W': 11.309, 'F1500W': 15.064}

TRIOS = {
    '3.3':  dict(filts=('F300M', 'F335M', 'F360M'),  contam='up'),
    '7.7':  dict(filts=('F560W', 'F770W', 'F1000W'), contam='low'),
    '11.3': dict(filts=('F1000W', 'F1130W', 'F1500W'), contam='up'),
}
K1 = {'3.3': (2.07, 0.30), '7.7': (4.33, 0.35), '11.3': (7.21, 0.92)}
K2 = {'3.3': (4.45, 0.39), '7.7': (5.84, 0.73), '11.3': (10.17, 1.24)}


def load_band(filt):
    path = os.path.join(ZP_DIR, f'{filt}_matchedF2100W_zp.fits')
    with fits.open(path) as hdu:
        sci = hdu['SCI'].data.astype(float)
        err = np.sqrt(hdu['ERR'].data.astype(float) ** 2 +
                      hdu['SCI'].header['ZPSYS'] ** 2)
    return sci, err


def linear_consub(f1, f2, f3, l1, l2, l3, e1, e2, e3):
    """No-contamination linear interpolation continuum."""
    beta = (l2 - l1) / (l3 - l1)
    con = (1 - beta) * f1 + beta * f3
    pah = f2 - con
    err = np.sqrt(((1 - beta) * e1) ** 2 + e2 ** 2 + (beta * e3) ** 2)
    return dict(pah=pah, con=con, pah_err=err)


bands = {}
for cfg in TRIOS.values():
    for f in cfg['filts']:
        if f not in bands:
            bands[f] = load_band(f)

# fiducial maps from step 01
fid = {b: fits.getdata(os.path.join(
    PAH_DIR, f'{TRIOS[b]["filts"][1]}_pah.fits'), 'PAH') for b in TRIOS}
fid_err = {b: fits.getdata(os.path.join(
    PAH_DIR, f'{TRIOS[b]["filts"][1]}_pah.fits'), 'PAH_ERR') for b in TRIOS}

METHODS = {}
for band, cfg in TRIOS.items():
    fl, fm, fu = cfg['filts']
    (f1, e1), (f2, e2), (f3, e3) = bands[fl], bands[fm], bands[fu]
    l1, l2, l3 = PIVOT[fl], PIVOT[fm], PIVOT[fu]
    func = get_pah_low if cfg['contam'] == 'low' else get_pah_up

    runs = {'lin': linear_consub(f1, f2, f3, l1, l2, l3, e1, e2, e3),
            'k2': func(f1, f2, f3, l1, l2, l3, K2[band][0],
                       e1, e2, e3, K2[band][1])}
    if band == '3.3':
        runs['sand16'] = func(f1, f2, f3, l1, l2, l3, 1.6, e1, e2, e3, 0.30)
    METHODS[band] = runs

    for meth, res in runs.items():
        hdus = fits.HDUList([fits.PrimaryHDU()])
        for name, arr in [('PAH', res['pah']), ('PAH_ERR', res['pah_err'])]:
            h = fits.getheader(os.path.join(PAH_DIR, f'{fm}_pah.fits'), 'PAH')
            h['EXTNAME'] = name
            h['CSMETHOD'] = meth
            hdus.append(fits.ImageHDU(arr.astype(np.float32), header=h))
        hdus.writeto(os.path.join(OUT_DIR, f'{fm}_pah_{meth}.fits'),
                     overwrite=True)

# ─── Statistics: ratio to fiducial on well-detected pixels ─
rows = []
for band in TRIOS:
    snr_fid = fid[band] / fid_err[band]
    good = snr_fid > 5
    for meth, res in METHODS[band].items():
        r = res['pah'][good] / fid[band][good]
        rows.append([band, meth, np.nanmedian(r),
                     np.nanpercentile(r, 16), np.nanpercentile(r, 84)])
        print(f'PAH {band:4s} {meth:7s}: PAH/PAH_k1 median = '
              f'{np.nanmedian(r):.3f}  (16-84%: {np.nanpercentile(r, 16):.3f}'
              f'-{np.nanpercentile(r, 84):.3f})  on {good.sum()} px SNR>5')

stats = Table(rows=rows, names=['band', 'method', 'median_ratio',
                                'p16', 'p84'])
stats.write(os.path.join(OUT_DIR, 'consub_method_stats.ecsv'),
            format='ascii.ecsv', overwrite=True)

# ─── Figure: maps of method/fiducial ratio + histograms ────
nb = len(TRIOS)
fig, axes = plt.subplots(2, nb, figsize=(6.2 * nb, 10),
                         gridspec_kw={'height_ratios': [3, 1.2]})
for j, band in enumerate(TRIOS):
    snr_fid = fid[band] / fid_err[band]
    good = snr_fid > 3
    rmap = np.where(good, METHODS[band]['lin']['pah'] / fid[band], np.nan)
    ax = axes[0, j]
    m = ax.imshow(rmap, origin='lower', cmap='RdBu_r', vmin=0.6, vmax=1.4)
    plt.colorbar(m, ax=ax, fraction=0.046)
    ax.set_title(f'PAH {band}: lin / k-method (SNR$_{{k1}}$>3)')
    ax.set_facecolor('0.85')
    ax.set_xticks([]); ax.set_yticks([])

    axh = axes[1, j]
    for meth, res in METHODS[band].items():
        r = res['pah'][snr_fid > 5] / fid[band][snr_fid > 5]
        axh.hist(r[np.isfinite(r)], bins=100, range=(0.5, 1.5),
                 histtype='step', lw=1.6, density=True, label=meth)
    axh.axvline(1, color='k', ls=':', lw=1)
    axh.set_xlabel(f'PAH {band} (method) / PAH {band} (k1)')
    axh.legend(fontsize=9)
    axh.set_yticks([])
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'consub_method_comparison.png'),
            dpi=115, bbox_inches='tight')
print('\nWrote', OUT_DIR)
