"""
08_working_set_analysis.py — downstream analysis over the ADOPTED sets
=======================================================================
SMC GO-5952 — con_sub module, step 08.

Propagates the adopted prescription working sets (WORKING_SETS.md:
3.3 = L20/W25/S23; 7.7 & 11.3 = k1/k2/lininterp/donnelly2025) through
the band-ratio and PAH-budget analysis: all 3x4x4 = 48 combinations,
yielding the systematic envelope of every headline number.

Outputs: prescriptions/working_set_combinations.ecsv
         prescriptions/combination_plane_working_set.png
         prescriptions/{band}_gallery_working_set.png (rotated style)

Usage: conda activate jwst && python 08_working_set_analysis.py
"""
import itertools
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astropy.visualization import simple_norm
from rotcrop_helper import rotcrop, common_bbox

P = os.path.expanduser('~/SMC_GO5952/products')
BASE = os.path.join(P, 'pah', 'prescriptions')
SETS = {'F335M': ['L20', 'W25', 'S23'],
        'F770W': ['tarantino_k1', 'tarantino_k2', 'lininterp',
                  'donnelly2025_F1000W'],
        'F1130W': ['tarantino_k1', 'tarantino_k2', 'lininterp',
                   'donnelly2025']}


def load(band, name):
    with fits.open(os.path.join(BASE, band, f'{band}_pah_{name}.fits')) as h:
        return h['PAH'].data.astype(float), h['PAH_ERR'].data.astype(float)


maps = {b: {n: load(b, n) for n in ns} for b, ns in SETS.items()}
f2100 = fits.getdata(os.path.join(P, 'matched', 'F2100W_matchedF2100W.fits'),
                     'SCI').astype(float)

# common mask: SNR>3 with the k1 references (fixed across combos)
mask = ((maps['F335M']['L20'][0] / maps['F335M']['L20'][1] > 3) &
        (maps['F770W']['tarantino_k1'][0] / maps['F770W']['tarantino_k1'][1] > 3) &
        (maps['F1130W']['tarantino_k1'][0] / maps['F1130W']['tarantino_k1'][1] > 3))
m21 = mask & (f2100 > 0.5)
print(f'common mask: {mask.sum()} px')


def med(x):
    return float(np.nanmedian(x))


rows = []
for a, b, c in itertools.product(*SETS.values()):
    A, B, C = maps['F335M'][a][0], maps['F770W'][b][0], maps['F1130W'][c][0]
    with np.errstate(invalid='ignore'):
        rows.append([a, b, c,
                     med(A[mask] / C[mask]), med(B[mask] / C[mask]),
                     med(A[mask] / B[mask]),
                     med((A + B + C)[mask]),
                     med(((A + B + C) / f2100)[m21])])
tab = Table(rows=rows, names=['p33', 'p77', 'p113', 'r33_113', 'r77_113',
                              'r33_77', 'sigma_pah', 'rpah_sigma_f2100'])
tab.write(os.path.join(BASE, 'working_set_combinations.ecsv'),
          format='ascii.ecsv', overwrite=True)
for col in ['r33_113', 'r77_113', 'r33_77', 'sigma_pah', 'rpah_sigma_f2100']:
    v = np.array(tab[col])
    print(f'{col:18s}: median {np.median(v):.3f}   envelope '
          f'[{v.min():.3f} - {v.max():.3f}]')

# ── plane figure (48 combos) ──
g = Table.read(os.path.join(P, 'pah', 'd21_model_ratios_v2.ecsv'),
               format='ascii.ecsv')
fig, ax = plt.subplots(figsize=(10.5, 8.5))
colors = {'lo': '#2166ac', 'st': '#1a9850', 'hi': '#b2182b'}
markers = {'sma': 'o', 'std': 's', 'lrg': '^'}
for row in g:
    ax.plot(row['r77_113'], row['r33_113'], marker=markers[row['size']],
            ms=11, color=colors[row['ion']], mec='k', mew=0.5, ls='none',
            zorder=3)
for size in ['sma', 'std', 'lrg']:
    s = g['size'] == size
    o = [list(g['ion'][s]).index(i) for i in ['lo', 'st', 'hi']]
    ax.plot(np.array(g['r77_113'][s])[o], np.array(g['r33_113'][s])[o],
            '-', color='0.5', lw=1, zorder=2)
fam = {'L20': '#4477aa', 'W25': '#66ccee', 'S23': '#ee6677'}
mk77 = {'tarantino_k1': 'o', 'tarantino_k2': 's', 'lininterp': 'D',
        'donnelly2025_F1000W': '^'}
for a in SETS['F335M']:
    s = tab['p33'] == a
    for b in SETS['F770W']:
        ss = s & (tab['p77'] == b)
        ax.scatter(tab['r77_113'][ss], tab['r33_113'][ss], s=42,
                   color=fam[a], marker=mk77[b], edgecolor='k', lw=0.4,
                   alpha=0.85, zorder=4)
star = tab[(tab['p33'] == 'L20') & (tab['p77'] == 'tarantino_k1') &
           (tab['p113'] == 'tarantino_k1')][0]
ax.plot(star['r77_113'], star['r33_113'], marker='*', ms=25, color='gold',
        mec='k', mew=1.2, zorder=6)
from matplotlib.lines import Line2D
handles = ([Line2D([], [], marker='o', ls='none', color=fam[a], ms=9,
                   label=f'3.3: {a}') for a in SETS['F335M']] +
           [Line2D([], [], marker=mk77[b], ls='none', color='0.4', ms=8,
                   label=f"7.7: {b.replace('tarantino_', 'k-').replace('2025_F1000W', '')}"
                   ) for b in SETS['F770W']] +
           [Line2D([], [], marker='*', ls='none', color='gold', mec='k',
                   ms=16, label='L20 / k1 / k1')])
ax.legend(handles=handles, fontsize=9, ncol=2, loc='upper left')
ax.set_xlabel('PAH 7.7 / 11.3  (ionization $\\rightarrow$)', fontsize=13)
ax.set_ylabel('PAH 3.3 / 11.3  (small sizes $\\rightarrow$)', fontsize=13)
ax.set_title('SMC median in the D21 plane — 48 adopted prescription '
             'combinations (systematic envelope of the working sets)',
             fontsize=12)
ax.set_xlim(0.35, 1.15)
ax.set_ylim(0.02, 0.20)
plt.tight_layout()
plt.savefig(os.path.join(BASE, 'combination_plane_working_set.png'),
            dpi=130, bbox_inches='tight')
plt.close()

# ── working-set galleries (rotated style) ──
ms_ = [np.isfinite(maps[b][ns[0]][0]) for b, ns in SETS.items()]
BBOX = common_bbox(np.stack(ms_ + [np.isfinite(f2100)]))
LBL = {'F335M': 'PAH 3.3 $\\mu$m', 'F770W': 'PAH 7.7 $\\mu$m',
       'F1130W': 'PAH 11.3 $\\mu$m'}
for band, ns in SETS.items():
    rmaps = {n: rotcrop(maps[band][n][0], BBOX) for n in ns}
    ens = np.nanmedian(np.stack(list(rmaps.values())), axis=0)
    bright = ens > np.nanpercentile(ens, 80)
    vmax = np.nanpercentile(ens, 99.4)
    n = len(ns)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 5.2))
    for ax, nm in zip(np.atleast_1d(axes), ns):
        ax.imshow(rmaps[nm], origin='lower', cmap='inferno',
                  norm=simple_norm(ens, 'asinh', vmin=0, vmax=vmax))
        ax.set_title(f"{nm}\nmedian = {np.nanmedian(rmaps[nm][bright]):.3f} "
                     "MJy/sr", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor('k')
    fig.suptitle(f'{LBL[band]} — adopted working set (identical absolute '
                 'scale)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE, f'{band}_gallery_working_set.png'),
                dpi=110, bbox_inches='tight')
    plt.close()
print('-> figures working set OK')
