"""
04b_donnelly.py — Donnelly+2025 empirical MIRI con-sub vs our k-method
======================================================================
SMC GO-5952 — con_sub module, step 04b.

Implements the spectroscopically calibrated prescription of
Donnelly et al. 2025 (arXiv:2501.19397; MIRI/MRS-calibrated on GOALS
LIRGs) for the 7.7 and 11.3 um PAH fluxes and compares the IN-BAND
PAH flux density (f_PAHband - g_cont * powerlaw continuum, MJy/sr)
to our k-method maps. Unlike the three-filter linear family, the
power-law continuum makes this a genuinely independent method: the
ratio map to the fiducial is NOT constant by construction.

Prescription (all degree-1 homogeneous -> valid on MJy/sr maps):
  cont_b = g_cont * f_blue^(1-alpha) * f_red^alpha,
  alpha = log(l_b/l_blue)/log(l_red/l_blue)  [pivot wavelengths]
  7.7 : Blue=F560W, Red=F1500W: alpha=0.31, g=0.68  (best, ~7%)
        Blue=F560W, Red=F1000W: alpha=0.53, g=0.91  (Tarantino's bench)
  11.3: two-branch on photometric silicate strength
        S_sil,phot = ln[F1000W/(F560W^0.42 F1500W^0.58)]
        S >= -0.6: cont = F1000W^(1-a) F1500W^a, a=0.31, g=1
        S <  -0.6: f10 -> 0.94 F1000W^1.20 F560W^-0.08 F1500W^-0.12

Outputs: products/pah/alternatives/donnelly_comparison.png
         products/pah/alternatives/{F770W,F1130W}_pah_donnelly.fits

Usage: conda activate jwst && python 04b_donnelly.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from astropy.io import fits

ZP  = os.path.expanduser('~/SMC_GO5952/products/matched')
PAH = os.path.expanduser('~/SMC_GO5952/products/pah')
OUT = os.path.join(PAH, 'alternatives')


def zp_map(f):
    return fits.getdata(os.path.join(ZP, f'{f}_matchedF2100W.fits'),
                        'SCI').astype(float)


f560, f770, f1000, f1130, f1500 = (zp_map(f) for f in
                                   ('F560W', 'F770W', 'F1000W',
                                    'F1130W', 'F1500W'))

# power-law expressions need positive fluxes
pos = (f560 > 0) & (f1000 > 0) & (f1500 > 0)

with np.errstate(invalid='ignore'):
    # ── 7.7 um ──
    cont77_best = np.where(pos, 0.68 * f560 ** (1 - 0.31) * f1500 ** 0.31,
                           np.nan)
    pah77_don = f770 - cont77_best
    cont77_trio = np.where(pos, 0.91 * f560 ** (1 - 0.53) * f1000 ** 0.53,
                           np.nan)
    pah77_don_trio = f770 - cont77_trio

    # ── 11.3 um, two-branch ──
    s_sil = np.where(pos, np.log(f1000 / (f560 ** 0.42 * f1500 ** 0.58)),
                     np.nan)
    a113 = 0.31
    cont_lo = f1000 ** (1 - a113) * f1500 ** a113
    f10c = 0.94 * f1000 ** 1.20 * f560 ** (-0.08) * f1500 ** (-0.12)
    cont_hi = f10c ** (1 - a113) * f1500 ** a113
    cont113 = np.where(s_sil < -0.6, cont_hi, cont_lo)
    pah113_don = f1130 - np.where(pos, cont113, np.nan)

frac_abs = np.nanmean(s_sil < -0.6) * 100
print(f'silicate branch: {frac_abs:.1f}% of pixels have S_sil,phot < -0.6')
print(f'S_sil,phot: median = {np.nanmedian(s_sil):.3f}, '
      f'16-84% = [{np.nanpercentile(s_sil,16):.3f}, '
      f'{np.nanpercentile(s_sil,84):.3f}]')

for name, arr in [('F770W_pah_donnelly', pah77_don),
                  ('F1130W_pah_donnelly', pah113_don)]:
    hdr = fits.getheader(os.path.join(ZP, 'F770W_matchedF2100W.fits'),
                         'SCI')
    fits.writeto(os.path.join(OUT, f'{name}.fits'),
                 arr.astype(np.float32), header=hdr, overwrite=True)

# ── comparison to fiducial k-method ──
stats_rows = []
comps = []
for band, fid_file, don, don_label in [
        ('7.7', 'F770W_pah.fits', pah77_don, 'B=F560W,R=F1500W'),
        ('7.7', 'F770W_pah.fits', pah77_don_trio, 'B=F560W,R=F1000W'),
        ('11.3', 'F1130W_pah.fits', pah113_don, 'two-branch,R=F1500W')]:
    with fits.open(os.path.join(PAH, fid_file)) as h:
        fid, ferr = h['PAH'].data.astype(float), h['PAH_ERR'].data.astype(float)
    good = (fid / ferr > 5) & np.isfinite(don)
    r = don[good] / fid[good]
    med, p16, p84 = (np.nanmedian(r), np.nanpercentile(r, 16),
                     np.nanpercentile(r, 84))
    print(f'PAH {band} Donnelly[{don_label}] / k-method: median = {med:.3f} '
          f'(16-84%: {p16:.3f}-{p84:.3f}) on {good.sum()} px')
    comps.append((band, don_label, don, fid, ferr, med))

fig, axes = plt.subplots(2, 3, figsize=(19, 11),
                         gridspec_kw={'height_ratios': [3, 1.2]})
for j, (band, lbl, don, fid, ferr, med) in enumerate(comps):
    good = fid / ferr > 3
    rmap = np.where(good & np.isfinite(don), don / fid, np.nan)
    ax = axes[0, j]
    m = ax.imshow(rmap, origin='lower', cmap='RdBu_r', vmin=0.6, vmax=1.4)
    plt.colorbar(m, ax=ax, fraction=0.046)
    ax.set_title(f'PAH {band}: Donnelly [{lbl}] / k-method')
    ax.set_facecolor('0.85'); ax.set_xticks([]); ax.set_yticks([])
    axh = axes[1, j]
    sel = (fid / ferr > 5) & np.isfinite(don)
    axh.hist((don / fid)[sel], bins=120, range=(0.5, 1.5),
             histtype='stepfilled', color='#666', alpha=0.85)
    axh.axvline(1, color='k', ls=':')
    axh.axvline(med, color='crimson', ls='--',
                label=f'médiane = {med:.3f}')
    axh.set_xlabel('Donnelly / k-method'); axh.set_yticks([])
    axh.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'donnelly_comparison.png'), dpi=115,
            bbox_inches='tight')
print('Wrote', os.path.join(OUT, 'donnelly_comparison.png'))
