"""
06_science_quicklook.py — first science products unique to GO-5952
===================================================================
SMC GO-5952 — con_sub module, step 06.

Products Tarantino+2025 could NOT make for Sextans A, computed here as
v0 quicklooks:

1. Sigma_PAH = PAH(3.3) + PAH(7.7) + PAH(11.3)  [SNR>3 in 7.7 and 11.3;
   3.3 added where SNR_3.3>2, else treated as 0 — it contributes ~7%]
2. R_PAH = Sigma_PAH / F2100W  (hot-dust-normalized PAH fraction proxy;
   Sextans A had no F2100W — this is our unique handle)
3. Pa-alpha map: F187N - continuum interpolated between F150W and F200W
   (v0: broadband contains the lines at the few-% level — flag)
4. H2 1-0 S(1) map: F212N - same continuum interpolation at 2.12 um
5. Binned trends of the band ratios vs Pa-alpha surface brightness —
   the pixel-based version of Tarantino's clump correlations with
   Halpha (she found rho_sp = 0.61 for 3.3/11.3, none for 7.7/11.3).

Outputs: products/science/  (FITS + PNG + stats)

Usage: conda activate jwst && python 06_science_quicklook.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from astropy.io import fits
from scipy import stats as sstats

ZP  = os.path.expanduser('~/SMC_GO5952/products/matched')
PAH = os.path.expanduser('~/SMC_GO5952/products/pah')
OUT = os.path.expanduser('~/SMC_GO5952/products/science')
os.makedirs(OUT, exist_ok=True)


def zp_map(f):
    return fits.getdata(os.path.join(ZP, f'{f}_matchedF2100W.fits'),
                        'SCI').astype(float)


def pah_map(f):
    # adopted 3.3 prescription: L20 (Meriem 2026-07-23)
    path = (os.path.join(PAH, 'prescriptions', 'F335M', 'F335M_pah_L20.fits')
            if f == 'F335M' else os.path.join(PAH, f'{f}_pah.fits'))
    with fits.open(path) as h:
        return h['PAH'].data.astype(float), h['PAH_ERR'].data.astype(float)


hdr = fits.getheader(os.path.join(ZP, 'F770W_matchedF2100W.fits'), 'SCI')

p33, e33 = pah_map('F335M')
p77, e77 = pah_map('F770W')
p113, e113 = pah_map('F1130W')

# ─── 1. Sigma_PAH ─────────────────────────────────────────
base = (p77 / e77 > 3) & (p113 / e113 > 3)
p33_c = np.where(p33 / e33 > 2, p33, 0.0)
sigma_pah = np.where(base, p33_c + p77 + p113, np.nan)
sigma_err = np.where(base, np.sqrt(e33 ** 2 + e77 ** 2 + e113 ** 2), np.nan)
fits.writeto(os.path.join(OUT, 'sigma_pah.fits'), sigma_pah.astype(np.float32),
             header=hdr, overwrite=True)

# ─── 2. R_PAH = Sigma_PAH / F2100W ────────────────────────
f2100 = zp_map('F2100W')
good21 = base & (f2100 > 0.5)          # avoid ratio blow-up at faint 21um
rpah = np.where(good21, sigma_pah / f2100, np.nan)
fits.writeto(os.path.join(OUT, 'rpah_sigma_over_f2100w.fits'),
             rpah.astype(np.float32), header=hdr, overwrite=True)
v = rpah[np.isfinite(rpah)]
print(f'Sigma_PAH/F2100W: median = {np.median(v):.3f} '
      f'(16-84%: {np.percentile(v,16):.3f}-{np.percentile(v,84):.3f}), '
      f'{v.size} px')

# ─── 3-4. Pa-alpha and H2 v0 maps ─────────────────────────
f150, f187, f200, f212 = (zp_map(f) for f in
                          ('F150W', 'F187N', 'F200W', 'F212N'))
l150, l187, l200, l212 = 1.501, 1.874, 1.990, 2.120
w = (l187 - l150) / (l200 - l150)
cont187 = (1 - w) * f150 + w * f200
paa = f187 - cont187
w2 = (l212 - l150) / (l200 - l150)      # mild extrapolation beyond F200W pivot
cont212 = (1 - w2) * f150 + w2 * f200
h2 = f212 - cont212
fits.writeto(os.path.join(OUT, 'paalpha_v0.fits'), paa.astype(np.float32),
             header=hdr, overwrite=True)
fits.writeto(os.path.join(OUT, 'h2_212_v0.fits'), h2.astype(np.float32),
             header=hdr, overwrite=True)

# ─── 5. Band ratios vs Pa-alpha ───────────────────────────
r33 = fits.getdata(os.path.join(PAH, 'ratio_33_113.fits'), 'RATIO')
r77 = fits.getdata(os.path.join(PAH, 'ratio_77_113.fits'), 'RATIO')

paa_pos = np.where(np.isfinite(paa) & (paa > 0), paa, np.nan)
results = {}
for name, ratio in [('3.3/11.3', r33), ('7.7/11.3', r77)]:
    sel = np.isfinite(ratio) & np.isfinite(paa_pos)
    x, y = np.log10(paa_pos[sel]), ratio[sel]
    rho, _ = sstats.spearmanr(x, y)
    # binned medians
    edges = np.percentile(x, np.linspace(0, 100, 13))
    xb, yb, yb16, yb84 = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum() > 100:
            xb.append(np.median(x[m])); yb.append(np.median(y[m]))
            yb16.append(np.percentile(y[m], 16))
            yb84.append(np.percentile(y[m], 84))
    results[name] = (x, y, np.array(xb), np.array(yb), np.array(yb16),
                     np.array(yb84), rho, sel.sum())
    print(f'{name} vs log Pa-alpha: Spearman rho = {rho:+.2f} '
          f'({sel.sum()} px)')

fig, axes = plt.subplots(1, 3, figsize=(19, 5.6))
ax = axes[0]
m = ax.imshow(rpah, origin='lower', cmap='viridis', vmin=0,
              vmax=np.nanpercentile(rpah, 98))
plt.colorbar(m, ax=ax, fraction=0.046)
ax.set_title('$\\Sigma$PAH / F2100W (proxy $R_{\\rm PAH}$)')
ax.set_facecolor('0.9'); ax.set_xticks([]); ax.set_yticks([])

for ax, (name, (x, y, xb, yb, y16, y84, rho, n)) in zip(axes[1:],
                                                        results.items()):
    hb = ax.hexbin(x, y, gridsize=90, cmap='Greys', bins='log',
                   extent=(np.percentile(x, 1), np.percentile(x, 99),
                           0, np.percentile(y, 99.5)))
    ax.errorbar(xb, yb, yerr=[yb - y16, y84 - yb], color='crimson',
                marker='o', ms=5, lw=1.8, capsize=2,
                label=f'médianes binées ($\\rho_{{sp}}$ = {rho:+.2f})')
    ax.set_xlabel('log Pa$\\alpha$ (v0, MJy/sr)')
    ax.set_ylabel(f'PAH {name}')
    ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'science_quicklook.png'), dpi=120,
            bbox_inches='tight')
print('Wrote', OUT)
