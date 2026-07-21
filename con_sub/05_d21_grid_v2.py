"""
05_d21_grid_v2.py — D21 grid with the SAME k-method con-sub as the data
========================================================================
SMC GO-5952 — con_sub module, step 05.

Improves on step 03 (which used direct synthetic photometry of the
PAH-only model spectrum) by treating the D21 models EXACTLY like the
data, as in Tarantino+2025 Sect. 3.6:

  1. synthetic photometry of the TOTAL dust spectrum (astrodust + PAH)
     in all trio filters -> f1, f2, f3 "observed" model fluxes;
  2. k derived from the D21 spectra themselves:
     k_D21 = synphot(PAH-only, PAH filter) / synphot(PAH-only,
     contaminated flank) — using the PAH+/PAH0 decomposition given in
     the model files (no continuum fitting needed);
  3. Eq. 5 continuum subtraction of the model f1/f2/f3 with k_D21.

The resulting grid ("v2") is what the DATA ratios should be compared
to when the data are continuum-subtracted with the k-method; the v1
grid (direct PAH photometry) is kept for comparison. The difference
between the grids quantifies the con-sub systematic on the model side.

Outputs: products/pah/d21_model_ratios_v2.ecsv
         products/pah/d21_ratio_plane_v2.png

Usage: conda activate jwst && python 05_d21_grid_v2.py
"""

import gzip
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.table import Table

D21_DIR = os.path.expanduser(
    '~/SMC_GO5952/models/Draine2021/BC03_Z0.0004_10Myr')
STPSF   = os.path.expanduser('~/data/stpsf-data')
PAH_DIR = os.path.expanduser('~/SMC_GO5952/products/pah')

INST = {'F300M': 'NIRCam', 'F335M': 'NIRCam', 'F360M': 'NIRCam',
        'F560W': 'MIRI', 'F770W': 'MIRI', 'F1000W': 'MIRI',
        'F1130W': 'MIRI', 'F1500W': 'MIRI'}
PIVOT = {'F300M': 2.996, 'F335M': 3.365, 'F360M': 3.621,
         'F560W': 5.635, 'F770W': 7.639, 'F1000W': 9.953,
         'F1130W': 11.309, 'F1500W': 15.064}
TRIOS = {
    '3.3':  dict(filts=('F300M', 'F335M', 'F360M'),  contam='up'),
    '7.7':  dict(filts=('F560W', 'F770W', 'F1000W'), contam='low'),
    '11.3': dict(filts=('F1000W', 'F1130W', 'F1500W'), contam='up'),
}
LOGU_RANGE = np.arange(0.0, 4.0, 0.5)
SIZES = ['sma', 'std', 'lrg']
IONS  = ['lo', 'st', 'hi']


def load_throughput(filt):
    t = Table.read(os.path.join(STPSF, INST[filt], 'filters',
                                f'{filt}_throughput.fits'))
    return np.asarray(t['WAVELENGTH'] / 1e4), np.asarray(t['THROUGHPUT'])


BANDPASS = {f: load_throughput(f) for f in INST}


def load_spectra(logu, ion, size):
    """-> wave [um], Fnu_pah, Fnu_total (arbitrary common units)."""
    name = f'pahspec.out_bc03_z0.0004_1e7_{logu:0.2f}_{ion}_{size}.gz'
    with gzip.open(os.path.join(D21_DIR, name), 'rt') as f:
        d = np.loadtxt(f, skiprows=7)
    wave = d[:, 0]
    fnu_pah = (d[:, 3] + d[:, 4]) * wave      # nu*P_nu -> P_nu ∝ *lambda
    fnu_tot = (d[:, 2] + d[:, 3] + d[:, 4]) * wave   # astrodust + PAHs
    return wave, fnu_pah, fnu_tot


def synphot(wave, fnu, filt):
    bw, bT = BANDPASS[filt]
    f = np.interp(bw, wave, fnu)
    w = bT / bw
    return np.trapz(f * w, bw) / np.trapz(w, bw)


def eq5(f1, f2, f3, l1, l2, l3, k, contam):
    beta = (l2 - l1) / (l3 - l1)
    y = f2 - (1 - beta) * f1 - beta * f3
    den = (k - beta) if contam == 'up' else (k - 1 + beta)
    return k / den * y


rows = []
for size in SIZES:
    for ion in IONS:
        acc = {b: {'v2': [], 'v1': [], 'k': []} for b in TRIOS}
        for logu in LOGU_RANGE:
            wave, fnu_pah, fnu_tot = load_spectra(logu, ion, size)
            for band, cfg in TRIOS.items():
                fl, fm, fu = cfg['filts']
                contam_filt = fu if cfg['contam'] == 'up' else fl
                k_d21 = (synphot(wave, fnu_pah, fm) /
                         synphot(wave, fnu_pah, contam_filt))
                fp2 = eq5(synphot(wave, fnu_tot, fl),
                          synphot(wave, fnu_tot, fm),
                          synphot(wave, fnu_tot, fu),
                          PIVOT[fl], PIVOT[fm], PIVOT[fu],
                          k_d21, cfg['contam'])
                acc[band]['v2'].append(fp2)
                acc[band]['v1'].append(synphot(wave, fnu_pah, fm))
                acc[band]['k'].append(k_d21)
        r33_113_v2 = np.array(acc['3.3']['v2']) / np.array(acc['11.3']['v2'])
        r77_113_v2 = np.array(acc['7.7']['v2']) / np.array(acc['11.3']['v2'])
        r33_113_v1 = np.array(acc['3.3']['v1']) / np.array(acc['11.3']['v1'])
        r77_113_v1 = np.array(acc['7.7']['v1']) / np.array(acc['11.3']['v1'])
        rows.append([size, ion,
                     np.mean(r33_113_v2), np.std(r33_113_v2),
                     np.mean(r77_113_v2), np.std(r77_113_v2),
                     np.mean(r33_113_v1), np.mean(r77_113_v1),
                     np.mean(acc['3.3']['k']), np.mean(acc['7.7']['k']),
                     np.mean(acc['11.3']['k'])])
        print(f"{size}/{ion}: v2 3.3/11.3 = {np.mean(r33_113_v2):.3f} "
              f"(v1 {np.mean(r33_113_v1):.3f}), v2 7.7/11.3 = "
              f"{np.mean(r77_113_v2):.3f} (v1 {np.mean(r77_113_v1):.3f}) | "
              f"k_D21 = {np.mean(acc['3.3']['k']):.2f}/"
              f"{np.mean(acc['7.7']['k']):.2f}/"
              f"{np.mean(acc['11.3']['k']):.2f}")

grid = Table(rows=rows, names=[
    'size', 'ion', 'r33_113', 'r33_113_std', 'r77_113', 'r77_113_std',
    'r33_113_v1', 'r77_113_v1', 'k33_d21', 'k77_d21', 'k113_d21'])
grid.meta['comment'] = [
    'v2: k-method con-sub applied to total D21 spectra with D21-derived',
    'k (Tarantino+2025 Sect. 3.6 approach); v1: direct PAH photometry.',
    'Averaged over logU=0-3.5.']
grid.write(os.path.join(PAH_DIR, 'd21_model_ratios_v2.ecsv'),
           format='ascii.ecsv', overwrite=True)

# ─── Figure: both grids over the data cloud ────────────────
r_size = fits.getdata(os.path.join(PAH_DIR, 'ratio_33_113.fits'), 'RATIO')
r_ion  = fits.getdata(os.path.join(PAH_DIR, 'ratio_77_113.fits'), 'RATIO')
good = np.isfinite(r_size) & np.isfinite(r_ion)
x, y = r_ion[good], r_size[good]

fig, ax = plt.subplots(figsize=(9.5, 8))
h = ax.hist2d(x, y, bins=150, range=[[0, 1.6], [0, 0.25]],
              cmap='Greys', norm=matplotlib.colors.LogNorm())
plt.colorbar(h[3], ax=ax, label='pixels (log)')

colors = {'lo': '#2166ac', 'st': '#1a9850', 'hi': '#b2182b'}
markers = {'sma': 'o', 'std': 's', 'lrg': '^'}
msize = {'sma': 8, 'std': 11, 'lrg': 14}
for tag, alpha, filled in [('v1', 0.35, False), ('v2', 1.0, True)]:
    xs = grid['r77_113'] if tag == 'v2' else grid['r77_113_v1']
    ys = grid['r33_113'] if tag == 'v2' else grid['r33_113_v1']
    for size in SIZES:
        s = grid['size'] == size
        order = [list(grid['ion'][s]).index(i) for i in IONS]
        ax.plot(np.array(xs[s])[order], np.array(ys[s])[order], '-',
                color='0.4', lw=1, alpha=alpha)
    for row, xx, yy in zip(grid, xs, ys):
        ax.plot(xx, yy, marker=markers[row['size']],
                ms=msize[row['size']], color=colors[row['ion']],
                mec='k', mew=0.5, alpha=alpha, ls='none',
                mfc=colors[row['ion']] if filled else 'none')

med = (np.median(x), np.median(y))
ax.plot(*med, marker='*', ms=26, color='gold', mec='k', mew=1.2, zorder=5)
ax.annotate('médiane SMC', med, textcoords='offset points',
            xytext=(12, -4), fontsize=11)
ax.set_xlabel('PAH 7.7 / 11.3  (ionisation →)', fontsize=13)
ax.set_ylabel('PAH 3.3 / 11.3  (petites tailles →)', fontsize=13)
ax.set_title('Grilles D21 : v2 = con-sub k comme les données (pleins), '
             'v1 = photométrie PAH directe (creux)', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(PAH_DIR, 'd21_ratio_plane_v2.png'), dpi=130,
            bbox_inches='tight')
print('\nWrote', os.path.join(PAH_DIR, 'd21_ratio_plane_v2.png'))
