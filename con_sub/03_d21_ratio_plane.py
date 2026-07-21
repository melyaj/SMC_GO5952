"""
03_d21_ratio_plane.py — PAH band-ratio diagnostic plane vs D21 models
======================================================================
SMC GO-5952 — con_sub module, step 03.

Compares the observed pixel-by-pixel PAH band ratios (step 02) to the
Draine+2021 model grids, following Tarantino+2025 Sect. 3.6 / Fig. 6:
BC03 Z=0.0004 10 Myr input radiation field, size distributions
sma/std/lrg (peak 3/4/5 A), ionization lo/st/hi (fion shifted x0.5 /
DL01 / x2), averaged over log U = 0-3.5 (band ratios insensitive to U
below logU~4).

Model band fluxes here = direct synthetic photometry of the PAH-only
model spectrum (PAH+ + PAH0) through the JWST bandpasses (stpsf-data
throughputs). NOTE: Tarantino+2025 instead runs the full k-method
continuum subtraction on the model total spectra with D21-derived k;
the difference is small for 7.7 and 11.3 (<~5%) and largest for 3.3
(the D21 spectra lack the 3.4 um aliphatic feature) — keep in mind for
the paper version.

Diagnostic plane: x = 7.7/11.3 (ionization), y = 3.3/11.3 (size).

Inputs : analysis_ready/pah/ratio_{33_113,77_113}.fits
         models/Draine2021/BC03_Z0.0004_10Myr/pahspec.out_*.gz
Outputs: analysis_ready/pah/d21_ratio_plane.png
         analysis_ready/pah/d21_model_ratios.ecsv

Usage: conda activate jwst && python 03_d21_ratio_plane.py
"""

import gzip
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.table import Table

# ─── Configuration ────────────────────────────────────────
D21_DIR = os.path.expanduser(
    '~/SMC_GO5952/models/Draine2021/BC03_Z0.0004_10Myr')
STPSF   = os.path.expanduser('~/data/stpsf-data')
PAH_DIR = os.path.expanduser('~/SMC_GO5952/analysis_ready/pah')

PAH_FILTERS = {'3.3': ('NIRCam', 'F335M'), '7.7': ('MIRI', 'F770W'),
               '11.3': ('MIRI', 'F1130W')}
LOGU_RANGE = np.arange(0.0, 4.0, 0.5)          # logU = 0 .. 3.5
SIZES = ['sma', 'std', 'lrg']
IONS  = ['lo', 'st', 'hi']


def load_throughput(inst, filt):
    """Filter curve -> (wave_um, T)."""
    t = Table.read(os.path.join(STPSF, inst, 'filters',
                                f'{filt}_throughput.fits'))
    return t['WAVELENGTH'] / 1e4, t['THROUGHPUT']


def load_pah_spectrum(logu, ion, size):
    """PAH-only model spectrum -> (wave_um, Fnu arbitrary units)."""
    name = f'pahspec.out_bc03_z0.0004_1e7_{logu:0.2f}_{ion}_{size}.gz'
    with gzip.open(os.path.join(D21_DIR, name), 'rt') as f:
        data = np.loadtxt(f, skiprows=7)
    wave = data[:, 0]                      # um
    nu_pnu_pah = data[:, 3] + data[:, 4]   # PAH+ + PAH0, nu*P_nu
    fnu = nu_pnu_pah * wave                # P_nu ∝ (nu P_nu)/nu ∝ *lambda
    return wave, fnu


def synphot(wave_um, fnu, band_wave, band_T):
    """Photon-weighted mean flux density in the bandpass."""
    f = np.interp(band_wave, wave_um, fnu)
    w = band_T / band_wave                  # T dlam/lam weighting
    return np.trapz(f * w, band_wave) / np.trapz(w, band_wave)


# ─── Model grid ratios ────────────────────────────────────
bands = {b: load_throughput(*iw) for b, iw in PAH_FILTERS.items()}

rows = []
for size in SIZES:
    for ion in IONS:
        r33_113, r77_113 = [], []
        for logu in LOGU_RANGE:
            wave, fnu = load_pah_spectrum(logu, ion, size)
            f = {b: synphot(wave, fnu, *bands[b]) for b in bands}
            r33_113.append(f['3.3'] / f['11.3'])
            r77_113.append(f['7.7'] / f['11.3'])
        rows.append([size, ion,
                     np.mean(r33_113), np.std(r33_113),
                     np.mean(r77_113), np.std(r77_113)])
        print(f'{size}/{ion}: 3.3/11.3 = {np.mean(r33_113):.3f} '
              f'+/- {np.std(r33_113):.3f}, 7.7/11.3 = '
              f'{np.mean(r77_113):.3f} +/- {np.std(r77_113):.3f}')

grid = Table(rows=rows, names=['size', 'ion', 'r33_113', 'r33_113_std',
                               'r77_113', 'r77_113_std'])
grid.meta['comment'] = [
    'D21 BC03_Z0.0004_10Myr PAH-only synthetic photometry ratios',
    'averaged over logU=0-3.5 (std over the logU range).']
grid.write(os.path.join(PAH_DIR, 'd21_model_ratios.ecsv'),
           format='ascii.ecsv', overwrite=True)

# ─── Observed pixel distribution ──────────────────────────
r_size = fits.getdata(os.path.join(PAH_DIR, 'ratio_33_113.fits'), 'RATIO')
r_ion  = fits.getdata(os.path.join(PAH_DIR, 'ratio_77_113.fits'), 'RATIO')
good = np.isfinite(r_size) & np.isfinite(r_ion)
x, y = r_ion[good], r_size[good]
print(f'\n{x.size} common pixels in the plane')

# ─── Figure ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9.5, 8))
h = ax.hist2d(x, y, bins=150, range=[[0, 1.6], [0, 0.25]],
              cmap='Greys', norm=matplotlib.colors.LogNorm())
plt.colorbar(h[3], ax=ax, label='pixels (log)')

colors = {'lo': '#2166ac', 'st': '#1a9850', 'hi': '#b2182b'}
markers = {'sma': 'o', 'std': 's', 'lrg': '^'}
msize = {'sma': 8, 'std': 11, 'lrg': 14}
# iso-size lines (vary ion) and iso-ion lines (vary size)
for size in SIZES:
    sel = grid[grid['size'] == size]
    sel = sel[[list(sel['ion']).index(i) for i in IONS]]
    ax.plot(sel['r77_113'], sel['r33_113'], '-', color='0.4', lw=1, zorder=3)
for ion in IONS:
    sel = grid[grid['ion'] == ion]
    sel = sel[[list(sel['size']).index(s) for s in SIZES]]
    ax.plot(sel['r77_113'], sel['r33_113'], '--', color=colors[ion],
            lw=1, zorder=3)
for row in grid:
    ax.errorbar(row['r77_113'], row['r33_113'],
                xerr=row['r77_113_std'], yerr=row['r33_113_std'],
                marker=markers[row['size']], ms=msize[row['size']],
                color=colors[row['ion']], mec='k', mew=0.5,
                zorder=4, capsize=2, ls='none')

med = (np.median(x), np.median(y))
ax.plot(*med, marker='*', ms=26, color='gold', mec='k', mew=1.2, zorder=5,
        label=f'mediane SMC ({med[0]:.2f}, {med[1]:.3f})')

from matplotlib.lines import Line2D
handles = [Line2D([], [], marker='*', ms=18, color='gold', mec='k', ls='none',
                  label=f'médiane SMC ({med[0]:.2f}, {med[1]:.3f})')]
handles += [Line2D([], [], marker=markers[s], color='0.3', ls='none',
                   ms=msize[s], label=f'taille {s} ({p} Å)')
            for s, p in zip(SIZES, [3, 4, 5])]
handles += [Line2D([], [], marker='s', color=colors[i], ls='none', ms=10,
                   label=f'ionisation {i}') for i in IONS]
ax.legend(handles=handles, loc='upper right', fontsize=10)

ax.set_xlabel('PAH 7.7 / 11.3  (ionisation →)', fontsize=13)
ax.set_ylabel('PAH 3.3 / 11.3  (petites tailles →)', fontsize=13)
ax.set_title('SMC GO-5952 pixels vs grilles D21 (BC03 Z=0.0004, 10 Myr, '
             'logU 0–3.5)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(PAH_DIR, 'd21_ratio_plane.png'), dpi=130,
            bbox_inches='tight')
print('Wrote', os.path.join(PAH_DIR, 'd21_ratio_plane.png'))
