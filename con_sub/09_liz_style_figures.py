"""
09_liz_style_figures.py — Tarantino+2025-style figure suite for the SMC
========================================================================
Fig.1 analog: field RGB + per-trio RGB zooms (continuum/PAH/continuum)
Fig.3 analog: continuum RGB vs PAH RGB + F2100W + Pa-alpha
Fig.7 analog: filter throughputs over a D21 spectrum
Fig.9 analog: radial profiles of the brightest PAH complex
Fig.2 analog: 14-filter SED of the Sigma_PAH peak region
Outputs -> products/pah/lizstyle_*.png
"""
import gzip, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astropy.visualization import make_lupton_rgb, simple_norm
from rotcrop_helper import rotcrop, common_bbox

P = os.path.expanduser('~/SMC_GO5952/products')
p33 = fits.getdata(os.path.join(P, 'pah/prescriptions/F335M/F335M_pah_L20.fits'), 'PAH').astype(float)
with fits.open(os.path.join(P, 'pah/F770W_pah.fits')) as h:
    p77, c77 = h['PAH'].data.astype(float), h['CON'].data.astype(float)
with fits.open(os.path.join(P, 'pah/F1130W_pah.fits')) as h:
    p113, c113 = h['PAH'].data.astype(float), h['CON'].data.astype(float)

def m(f):
    return fits.getdata(os.path.join(P, f'matched/{f}_matchedF2100W.fits'), 'SCI').astype(float)

paa = fits.getdata(os.path.join(P, 'science/paalpha_v0.fits')).astype(float)
sig = fits.getdata(os.path.join(P, 'science/sigma_pah.fits')).astype(float)
BBOX = common_bbox(np.stack([np.isfinite(p77)]))

def ch(img, pmax=99.6):
    c = np.nan_to_num(img - np.nanmedian(img))
    return np.clip(c / np.nanpercentile(c, pmax), 0, 1)

def rgb(r, g, b, Q=9, stretch=0.55):
    return make_lupton_rgb(ch(r), ch(g), ch(b), Q=Q, stretch=stretch)

# ── Fig.1 analog ──
zy, zx, zh = 620, 520, 180          # brightest complex (approx, grid px)
peak = np.unravel_index(np.nanargmax(np.where(np.isfinite(sig), sig, -1)), sig.shape)
zy, zx = peak
fig = plt.figure(figsize=(19, 10.5))
gs = fig.add_gridspec(2, 3, height_ratios=[1.55, 1])
ax0 = fig.add_subplot(gs[0, :])
big = rgb(rotcrop(m('F1000W'), BBOX), rotcrop(m('F335M'), BBOX), rotcrop(m('F150W'), BBOX))
ax0.imshow(big, origin='lower')
ax0.set_title('SMC GO-5952 — R: F1000W, G: F335M, B: F150W', fontsize=13)
ax0.set_xticks([]); ax0.set_yticks([])
trios = [(('F360M', 'F335M', 'F300M'), '3.3 $\\mu$m trio\nR: F360M  G: F335M  B: F300M'),
         (('F1000W', 'F770W', 'F560W'), '7.7 $\\mu$m trio\nR: F1000W  G: F770W  B: F560W'),
         (('F1500W', 'F1130W', 'F1000W'), '11.3 $\\mu$m trio\nR: F1500W  G: F1130W  B: F1000W')]
for j, ((r_, g_, b_), title) in enumerate(trios):
    ax = fig.add_subplot(gs[1, j])
    cut = lambda f: m(f)[zy-zh:zy+zh, zx-zh:zx+zh]
    ax.imshow(rgb(cut(r_), cut(g_), cut(b_), stretch=0.7), origin='lower')
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle('Field composite and PAH-trio zooms (green excess = PAH emission) '
             f'— zoom {2*zh*0.11:.0f}\" on the brightest complex', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(P, 'pah', 'lizstyle_fig1_composites.png'), dpi=110,
            bbox_inches='tight')
plt.close()

# ── Fig.3 analog ──
c33 = m('F335M') - p33
fig, axes = plt.subplots(2, 2, figsize=(14.5, 15))
panels = [
    (rgb(rotcrop(c113, BBOX), rotcrop(c77, BBOX), rotcrop(c33, BBOX)),
     'Continuum RGB — R: 11.3 cont, G: 7.7 cont, B: 3.3 cont'),
    (rgb(rotcrop(p113, BBOX), rotcrop(p77, BBOX), rotcrop(p33, BBOX)),
     'Continuum-subtracted PAH RGB — R: 11.3, G: 7.7, B: 3.3'),
    (rotcrop(m('F2100W'), BBOX), 'F2100W hot dust'),
    (rotcrop(paa, BBOX), 'Pa$\\alpha$ (F187N, v0) — ionized gas')]
for ax, (img, title) in zip(axes.ravel(), panels):
    if img.ndim == 3:
        ax.imshow(img, origin='lower')
    else:
        ax.imshow(img, origin='lower', cmap='afmhot',
                  norm=simple_norm(img, 'asinh', vmin=0,
                                   vmax=np.nanpercentile(img, 99.5)))
    ax.set_title(title, fontsize=13)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor('k')
fig.suptitle('Continuum vs PAH emission, hot dust and ionized gas '
             '(Tarantino+2025 Fig. 3 layout; Pa$\\alpha$ replaces H$\\alpha$)',
             fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(P, 'pah', 'lizstyle_fig3_rgb.png'), dpi=110,
            bbox_inches='tight')
plt.close()

# ── Fig.7 analog ──
D21 = os.path.expanduser('~/SMC_GO5952/models/Draine2021/BC03_Z0.0004_10Myr')
d = np.loadtxt(gzip.open(os.path.join(
    D21, 'pahspec.out_bc03_z0.0004_1e7_1.00_st_std.gz'), 'rt'), skiprows=7)
wave, ftot = d[:, 0], (d[:, 2] + d[:, 3] + d[:, 4]) * d[:, 0]
STPSF = os.path.expanduser('~/data/stpsf-data')
INST = {'F300M': 'NIRCam', 'F335M': 'NIRCam', 'F360M': 'NIRCam',
        'F444W': 'NIRCam', 'F560W': 'MIRI', 'F770W': 'MIRI',
        'F1000W': 'MIRI', 'F1130W': 'MIRI', 'F1500W': 'MIRI', 'F2100W': 'MIRI'}
ROLE = {'F335M': '#1a9850', 'F770W': '#1a9850', 'F1130W': '#1a9850',
        'F300M': '#c51b7d', 'F1000W': '#c51b7d',
        'F360M': '#e6a700', 'F560W': '#e6a700', 'F1500W': '#e6a700',
        'F444W': '#888888', 'F2100W': '#888888'}
fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                             gridspec_kw={'height_ratios': [1.3, 1]})
a1.plot(wave, ftot / np.nanmax(ftot[(wave > 2.5) & (wave < 22)]), color='k', lw=1.2)
a1.set_ylabel('normalized $F_\\nu$')
a1.set_title('D21 model spectrum (BC03 Z=0.0004, 10 Myr, logU=1, st/std) '
             'with the GO-5952 filter set', fontsize=12)
a1.set_xlim(2.5, 23); a1.set_ylim(0, 1.1)
for f, inst in INST.items():
    t = Table.read(f'{STPSF}/{inst}/filters/{f}_throughput.fits')
    w, T = np.asarray(t['WAVELENGTH'])/1e4, np.asarray(t['THROUGHPUT'])
    a2.fill_between(w, 0, T, color=ROLE[f], alpha=0.45)
    a2.text(np.average(w, weights=T), np.nanmax(T)+0.02, f, fontsize=8,
            ha='center', color=ROLE[f])
a2.set_xlabel('wavelength ($\\mu$m)'); a2.set_ylabel('transmission')
a2.set_ylim(0, 0.85)
from matplotlib.patches import Patch
a2.legend(handles=[Patch(color='#1a9850', label='PAH filter'),
                   Patch(color='#c51b7d', label='clean continuum'),
                   Patch(color='#e6a700', label='PAH-contaminated continuum'),
                   Patch(color='#888888', label='other')], fontsize=9, loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(P, 'pah', 'lizstyle_fig7_filters.png'), dpi=120,
            bbox_inches='tight')
plt.close()

# ── Fig.9 analog: radial profiles of the brightest complex ──
def radial(img, cy, cx, rmax_px=364):
    yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]
    r = np.hypot(xx-cx, yy-cy) * 0.11
    rb = np.arange(0, rmax_px*0.11, 1.0)
    prof = np.array([np.nanmedian(img[(r >= a) & (r < a+1.0)]) for a in rb])
    return rb + 0.5, prof

fig, ax = plt.subplots(figsize=(10, 6.5))
for img, lab, c in [(p113, 'PAH 11.3 $\\mu$m', '#4477aa'),
                    (c113, '11.3 $\\mu$m continuum', '#1a9850'),
                    (m('F2100W'), 'F2100W hot dust', '#e6a700'),
                    (paa, 'Pa$\\alpha$ (v0)', '#cc3311')]:
    rb, prof = radial(img, zy, zx)
    prof = prof / np.nansum(prof)
    ax.plot(rb, prof, color=c, lw=2, label=lab)
ax.set_xlabel('radius from $\\Sigma$PAH peak (arcsec)', fontsize=12)
ax.set_ylabel('normalized azimuthal-median profile', fontsize=12)
ax.set_title(f'Radial profiles of the brightest PAH complex '
             f'(center: grid px {zx}, {zy}; 1\" = 0.3 pc)', fontsize=12)
ax.legend(fontsize=11)
ax.set_xlim(0, 40)
plt.tight_layout()
plt.savefig(os.path.join(P, 'pah', 'lizstyle_fig9_radial.png'), dpi=125,
            bbox_inches='tight')
plt.close()

# ── Fig.2 analog: SED of the Sigma_PAH peak ──
FILT = ['F150W', 'F187N', 'F200W', 'F212N', 'F300M', 'F335M', 'F360M',
        'F444W', 'F560W', 'F770W', 'F1000W', 'F1130W', 'F1500W', 'F2100W']
PIV = {'F150W': 1.501, 'F187N': 1.874, 'F200W': 1.990, 'F212N': 2.120,
       'F300M': 2.996, 'F335M': 3.365, 'F360M': 3.621, 'F444W': 4.404,
       'F560W': 5.635, 'F770W': 7.639, 'F1000W': 9.953, 'F1130W': 11.309,
       'F1500W': 15.064, 'F2100W': 20.795}
PAHF = {'F335M', 'F770W', 'F1130W'}
CONTF = {'F300M', 'F360M', 'F560W', 'F1000W', 'F1500W'}
ap = 14   # 1.5" aperture radius
yy, xx = np.mgrid[:p77.shape[0], :p77.shape[1]]
sel = np.hypot(xx-zx, yy-zy) < ap
fig, ax = plt.subplots(figsize=(10.5, 6.5))
for f in FILT:
    val = np.nanmean(m(f)[sel])
    color = ('#1a9850' if f in PAHF else '#c51b7d' if f in CONTF else '0.4')
    ax.plot(PIV[f], val, 'o', ms=9, color=color, mec='k', mew=0.5)
    ax.annotate(f, (PIV[f], val), textcoords='offset points', xytext=(0, 9),
                fontsize=8, ha='center', color=color)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('wavelength ($\\mu$m)', fontsize=12)
ax.set_ylabel('mean surface brightness (MJy/sr)', fontsize=12)
ax.set_title(f'SED of the brightest PAH complex (r=1.5\" aperture at the '
             '$\\Sigma$PAH peak)\ngreen: PAH filters — magenta: continuum '
             'filters — grey: other', fontsize=12)
from matplotlib.ticker import ScalarFormatter
ax.xaxis.set_major_formatter(ScalarFormatter())
plt.tight_layout()
plt.savefig(os.path.join(P, 'pah', 'lizstyle_fig2_sed.png'), dpi=125,
            bbox_inches='tight')
plt.close()
print('5 figures liz-style ->', os.path.join(P, 'pah'))
