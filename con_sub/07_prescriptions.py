"""
07_prescriptions.py — ALL published continuum-subtraction prescriptions
========================================================================
SMC GO-5952 — con_sub module, step 07. THE core methodology framework:
apply every published prescription for each PAH band, pixel by pixel,
with propagated uncertainties, organized for systematic comparison
(the heart of the methodology paper).

3.3 um (F335M) — all of the form  PAH = a*F335M - b*F300M - c*F360M:
  lai2020        a=1.000 b=0.350 c=0.650   (Lai+2020 Eq.1-2)
  sandstrom2023  a=1.684 b=0.589 c=1.094   (=1.684*(F335M-0.35F300M-0.65F360M))
  whitcomb2025   a=1.200 b=0.456 c=0.744   (=1.20*(F335M-0.38F300M-0.62F360M))
  bolatto2024    a=1.520 b=0.638 c=0.882   (=1.52*(F335M-0.42F300M-0.58F360M))
  tarantino_k1   a=1.399 b=0.574 c=0.826   (k=2.07, beta=0.590 pivots)
  tarantino_k2   a=1.153 b=0.473 c=0.680   (k=4.45)
  lininterp      a=1.000 b=0.410 c=0.590   (pivot interpolation, no contam.)

7.7 um (F770W):
  tarantino_k1 / tarantino_k2 / lininterp  (k-method, k=4.33 / 5.84 / inf)
  donnelly2025_F1500W   cont = 0.68 * F560W^0.69 * F1500W^0.31  (best, ~7%)
  donnelly2025_F1000W   cont = 0.91 * F560W^0.47 * F1000W^0.53
  chown2025             PAH = F770W - 1.14*F1000W  (PDR regime only)

11.3 um (F1130W):
  tarantino_k1 / tarantino_k2 / lininterp  (k=7.21 / 10.17 / inf)
  donnelly2025          two-branch silicate (S_sil,phot threshold -0.6)
  chown2025_F1000W      PAH = F1130W - 1.00*F1000W
  chown2025_F1500W      PAH = F1130W - 0.29*F1500W

Notes: Chown+2025 beta offsets (fitted to the very bright Orion Bar)
are dropped — not transferable to faint extragalactic fields (their
paper's own caveat). All expressions are degree-1 homogeneous ->
valid in MJy/sr. Errors: full linear propagation; power laws to first
order. k-method errors include k_err (exact derivatives, see step 01).

Outputs: products/pah/prescriptions/{F335M,F770W,F1130W}/
           {band}_pah_{method}.fits  (PAH, PAH_ERR; METHOD/REF/FORMULA
           in headers)
         products/pah/prescriptions/prescriptions_summary.ecsv
         products/pah/prescriptions/{band}_methods_comparison.png

Usage: conda activate jwst && python 07_prescriptions.py
"""

import os
import types
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.table import Table

MATCHED = os.path.expanduser('~/SMC_GO5952/products/matched')
OUT = os.path.expanduser('~/SMC_GO5952/products/pah/prescriptions')

PIVOT = {'F300M': 2.996, 'F335M': 3.365, 'F360M': 3.621, 'F444W': 4.404,
         'F560W': 5.635, 'F770W': 7.639, 'F1000W': 9.953,
         'F1130W': 11.309, 'F1500W': 15.064}

# k-method exact functions from step 01 (corrected error propagation)
src = open(os.path.expanduser(
    '~/SMC_GO5952/pipeline/con_sub/01_consub_kmethod.py')).read()
mod = types.ModuleType('kf')
mod.np = np
exec(src[src.index('def get_pah_low'):src.index('# ─── I/O helpers')],
     mod.__dict__)
get_pah_low, get_pah_up = mod.get_pah_low, mod.get_pah_up


def load(f):
    p = os.path.join(MATCHED, f'{f}_matchedF2100W.fits')
    with fits.open(p) as h:
        sci = h['SCI'].data.astype(float)
        err = np.sqrt(h['ERR'].data.astype(float) ** 2 +
                      h['SCI'].header['ZPSYS'] ** 2)
        hdr = h['SCI'].header
    return sci, err, hdr


bands = {f: load(f) for f in PIVOT}


def lincombo(a, b, c, f2, fb, fr, e2, eb, er):
    """PAH = a*f2 - b*fb - c*fr with linear error propagation."""
    pah = a * f2 - b * fb - c * fr
    err = np.sqrt((a * e2) ** 2 + (b * eb) ** 2 + (c * er) ** 2)
    return pah, err


def kmethod(trio, contam, k, kerr):
    (f1, e1, _), (f2, e2, _), (f3, e3, _) = (bands[t] for t in trio)
    l1, l2, l3 = (PIVOT[t] for t in trio)
    func = get_pah_low if contam == 'low' else get_pah_up
    r = func(f1, f2, f3, l1, l2, l3, k, e1, e2, e3, kerr)
    return r['pah'], r['pah_err']


def powerlaw_cont(g, fb, alpha, fr, eb, er):
    """cont = g * fb^(1-alpha) * fr^alpha, first-order errors."""
    with np.errstate(invalid='ignore'):
        pos = (fb > 0) & (fr > 0)
        cont = np.where(pos, g * fb ** (1 - alpha) * fr ** alpha, np.nan)
        econt = np.abs(cont) * np.sqrt(((1 - alpha) * eb / fb) ** 2 +
                                       (alpha * er / fr) ** 2)
    return cont, econt


# ─── method registry ──────────────────────────────────────
METHODS = {'F335M': {}, 'F770W': {}, 'F1130W': {}}

# 3.3 um — unified form of Elyajouri et al. Paper I, Eq. 3 / Table 2:
# PAH = alpha * [F335M - (1-beta)*F300M - beta*F360M]
# ADOPTED F335M SET (Meriem, 2026-07-23):
#   L20  — aromatic 3.3 (best when 3.4 is faint)
#   W25  — adopted compromise (3.3+3.4)
#   S23  — total PAH-dust-correlated in-band emission
#   T25  — PDRs4All PAHfit k=2.07 (as used in Paper I; NOT the D21 k2)
#   B24  — F300M-recalibrated form (per W25)
#   E26  — Paper I recommendation: F300M+F335M+F444W, alpha=1.25+/-0.06
#          (combined 3.3+3.4-3.46; most universal, Table 4)
for name, alpha, beta, blue, red, ref in [
        ('L20', 1.00, 0.650, 'F300M', 'F360M', 'Lai et al. 2020 (aromatic 3.3)'),
        ('W25', 1.20, 0.620, 'F300M', 'F360M', 'Whitcomb et al. 2025 (adopted mean)'),
        ('S23', 1.68, 0.650, 'F300M', 'F360M', 'Sandstrom et al. 2023 (PAH-correlated total)'),
        ('T25', 1.40, 0.590, 'F300M', 'F360M', 'Tarantino et al. 2025 (PDRs4All PAHfit k=2.07)'),
        ('B24', 1.52, 0.580, 'F300M', 'F360M', 'Bolatto et al. 2024 (F300M-recalibrated)'),
        ('E26', 1.25, 0.264, 'F300M', 'F444W', 'Elyajouri et al. Paper I (F444W anchor, combined)')]:
    METHODS['F335M'][name] = dict(
        kind='lin', a=alpha, b=alpha * (1 - beta), c=alpha * beta,
        blue=blue, red=red,
        formula=f'{alpha:.2f}*(F335M - {1-beta:.3f}*{blue} - {beta:.3f}*{red})',
        ref=ref)
# 7.7 um
METHODS['F770W'] = {
    'tarantino_k1': dict(kind='k', trio=('F560W', 'F770W', 'F1000W'),
                         contam='low', k=4.33, kerr=0.35,
                         formula='k-method k=4.33+/-0.35',
                         ref='Tarantino et al. 2025 (k1, PAHFIT)'),
    'tarantino_k2': dict(kind='k', trio=('F560W', 'F770W', 'F1000W'),
                         contam='low', k=5.84, kerr=0.73,
                         formula='k-method k=5.84+/-0.73',
                         ref='Tarantino et al. 2025 (k2, D21)'),
    'lininterp': dict(kind='lin770', a=1.0, wA=0.536, wB=0.464,
                      formula='F770W - (0.536*F560W + 0.464*F1000W)',
                      ref='pivot interpolation'),
    'donnelly2025_F1500W': dict(kind='pl', band='F770W', g=0.68,
                                blue='F560W', red='F1500W', alpha=0.31,
                                formula='F770W - 0.68*F560W^0.69*F1500W^0.31',
                                ref='Donnelly et al. 2025 (best anchors)'),
    'donnelly2025_F1000W': dict(kind='pl', band='F770W', g=0.91,
                                blue='F560W', red='F1000W', alpha=0.53,
                                formula='F770W - 0.91*F560W^0.47*F1000W^0.53',
                                ref='Donnelly et al. 2025'),
    'chown2025': dict(kind='chown', band='F770W', bsub='F1000W', coef=1.14,
                      formula='F770W - 1.14*F1000W (PDR regime; no offset)',
                      ref='Chown et al. 2025 (PDRs4All XIII)'),
}

# 11.3 um
METHODS['F1130W'] = {
    'tarantino_k1': dict(kind='k', trio=('F1000W', 'F1130W', 'F1500W'),
                         contam='up', k=7.21, kerr=0.92,
                         formula='k-method k=7.21+/-0.92',
                         ref='Tarantino et al. 2025 (k1, polyfit)'),
    'tarantino_k2': dict(kind='k', trio=('F1000W', 'F1130W', 'F1500W'),
                         contam='up', k=10.17, kerr=1.24,
                         formula='k-method k=10.17+/-1.24',
                         ref='Tarantino et al. 2025 (k2, D21)'),
    'lininterp': dict(kind='lin1130', a=1.0, wA=0.735, wB=0.265,
                      formula='F1130W - (0.735*F1000W + 0.265*F1500W)',
                      ref='pivot interpolation'),
    'donnelly2025': dict(kind='don113',
                         formula='two-branch silicate (S_sil threshold -0.6)',
                         ref='Donnelly et al. 2025'),
    'chown2025_F1000W': dict(kind='chown', band='F1130W', bsub='F1000W',
                             coef=1.00,
                             formula='F1130W - 1.00*F1000W (no offset)',
                             ref='Chown et al. 2025 (PDRs4All XIII)'),
    'chown2025_F1500W': dict(kind='chown', band='F1130W', bsub='F1500W',
                             coef=0.29,
                             formula='F1130W - 0.29*F1500W (no offset)',
                             ref='Chown et al. 2025 (PDRs4All XIII)'),
}


def compute(band, name, m):
    if m['kind'] == 'k':
        return kmethod(m['trio'], m['contam'], m['k'], m['kerr'])
    if m['kind'] == 'lin':
        f2, e2, _ = bands['F335M']
        fb, eb, _ = bands[m.get('blue', 'F300M')]
        fr, er, _ = bands[m.get('red', 'F360M')]
        return lincombo(m['a'], m['b'], m['c'], f2, fb, fr, e2, eb, er)
    if m['kind'] == 'lin770':
        f2, e2, _ = bands['F770W']
        fb, eb, _ = bands['F560W']
        fr, er, _ = bands['F1000W']
        return lincombo(1.0, m['wA'], m['wB'], f2, fb, fr, e2, eb, er)
    if m['kind'] == 'lin1130':
        f2, e2, _ = bands['F1130W']
        fb, eb, _ = bands['F1000W']
        fr, er, _ = bands['F1500W']
        return lincombo(1.0, m['wA'], m['wB'], f2, fb, fr, e2, eb, er)
    if m['kind'] == 'pl':
        f2, e2, _ = bands[m['band']]
        fb, eb, _ = bands[m['blue']]
        fr, er, _ = bands[m['red']]
        cont, ec = powerlaw_cont(m['g'], fb, m['alpha'], fr, eb, er)
        return f2 - cont, np.sqrt(e2 ** 2 + ec ** 2)
    if m['kind'] == 'chown':
        f2, e2, _ = bands[m['band']]
        fb, eb, _ = bands[m['bsub']]
        return f2 - m['coef'] * fb, np.sqrt(e2 ** 2 + (m['coef'] * eb) ** 2)
    if m['kind'] == 'don113':
        f2, e2, _ = bands['F1130W']
        f5, e5, _ = bands['F560W']
        f10, e10, _ = bands['F1000W']
        f15, e15, _ = bands['F1500W']
        with np.errstate(invalid='ignore'):
            pos = (f5 > 0) & (f10 > 0) & (f15 > 0)
            s_sil = np.where(pos, np.log(f10 / (f5**0.42 * f15**0.58)), np.nan)
            a = 0.31
            c_lo, e_lo = powerlaw_cont(1.0, f10, a, f15, e10, e15)
            f10c = 0.94 * f10**1.20 * f5**(-0.08) * f15**(-0.12)
            e10c = np.abs(f10c) * np.sqrt((1.20*e10/f10)**2 + (0.08*e5/f5)**2
                                          + (0.12*e15/f15)**2)
            c_hi, e_hi = powerlaw_cont(1.0, f10c, a, f15, e10c, e15)
            cont = np.where(s_sil < -0.6, c_hi, c_lo)
            ec = np.where(s_sil < -0.6, e_hi, e_lo)
        return f2 - cont, np.sqrt(e2 ** 2 + ec ** 2)
    raise ValueError(m['kind'])


rows = []
FIDUCIALS = {'F335M': 'T25', 'F770W': 'tarantino_k1', 'F1130W': 'tarantino_k1'}
for band in METHODS:
    os.makedirs(os.path.join(OUT, band), exist_ok=True)
    results = {}
    for name, m in METHODS[band].items():
        pah, err = compute(band, name, m)
        results[name] = (pah, err)
        hdr = bands[band][2].copy()
        hdr['METHOD'] = name
        hdr['MREF'] = (m['ref'][:68], 'prescription reference')
        hdr['MFORMULA'] = (m['formula'][:68], 'prescription formula')
        hdus = fits.HDUList([fits.PrimaryHDU()])
        for ext, arr in [('PAH', pah), ('PAH_ERR', err)]:
            h = hdr.copy()
            h['EXTNAME'] = ext
            hdus.append(fits.ImageHDU(arr.astype(np.float32), header=h))
        hdus.writeto(os.path.join(OUT, band, f'{band}_pah_{name}.fits'),
                     overwrite=True)
    # neutral baseline: pixelwise median across all prescriptions
    import numpy as _np
    _stack = _np.stack([results[n][0] for n in METHODS[band]])
    fid = _np.nanmedian(_stack, axis=0)
    fide = results[list(METHODS[band])[0]][1]
    good = fid / fide > 5
    for name in METHODS[band]:
        pah, err = results[name]
        with np.errstate(invalid='ignore'):
            r = pah[good] / fid[good]
        med = float(np.nanmedian(r))
        p16, p84 = (float(np.nanpercentile(r, 16)),
                    float(np.nanpercentile(r, 84)))
        snr3 = float(np.nanmean((pah / err > 3)[np.isfinite(pah / err)]))
        rows.append([band, name, med, p16, p84, 100 * snr3,
                     METHODS[band][name]['ref']])
        print(f"{band} {name:22s}: /fiducial = {med:.3f} "
              f"({p16:.3f}-{p84:.3f})  SNR>3: {100*snr3:.0f}%")

    # comparison figure: histogram of method/fiducial + map of most deviant
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for name in METHODS[band]:
        if name == FIDUCIAL:
            continue
        r = results[name][0][good] / fid[good]
        axes[0].hist(r[np.isfinite(r)], bins=120, range=(0.3, 1.7),
                     histtype='step', lw=1.6, density=True, label=name)
    axes[0].axvline(1, color='k', ls=':')
    axes[0].set_xlabel(f'PAH({band}, methode) / PAH(k1)')
    axes[0].legend(fontsize=8)
    axes[0].set_yticks([])
    devs = {n: abs(np.nanmedian(results[n][0][good] / fid[good]) - 1)
            for n in METHODS[band] if n != FIDUCIAL}
    worst = max(devs, key=devs.get)
    with np.errstate(invalid='ignore'):
        rmap = np.where(good, results[worst][0] / fid, np.nan)
    im = axes[1].imshow(rmap, origin='lower', cmap='RdBu_r',
                        vmin=0.5, vmax=1.5)
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    axes[1].set_title(f'{worst} / k1 (carte)')
    axes[1].set_xticks([]); axes[1].set_yticks([])
    axes[1].set_facecolor('0.85')
    fig.suptitle(f'{band}: comparaison des prescriptions (px SNR>5)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, f'{band}_methods_comparison.png'),
                dpi=120, bbox_inches='tight')
    plt.close()

tab = Table(rows=rows, names=['band', 'method', 'median_vs_ensemble', 'p16',
                              'p84', 'pct_snr3', 'reference'])
tab.meta['comment'] = [
    'All published continuum-subtraction prescriptions applied to the',
    'SMC GO-5952 matched maps; ratios vs the pixelwise ENSEMBLE MEDIAN',
    'of all prescriptions (no fiducial), on SNR>5 pixels. Chown+2025',
    'beta offsets dropped (Orion-calibrated).']
tab.write(os.path.join(OUT, 'prescriptions_summary.ecsv'),
          format='ascii.ecsv', overwrite=True)
print('\n->', OUT)
