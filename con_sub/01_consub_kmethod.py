"""
01_consub_kmethod.py — k-method continuum subtraction of the PAH filters
=========================================================================
SMC GO-5952 — con_sub module, step 01.

Implements the continuum subtraction of Tarantino+2025 (Sect. 3.1.4,
Eq. 5), adapted from her sextansA_PAH_paper code (k_eq_with_err.py).
Each PAH filter trio: f1/f2/f3 with f2 the PAH filter, one flanking
filter clean continuum, the other contaminated by PAH complexes with
f_p(contaminated) = f_p2 / k.

Adopted k values = k1 of Tarantino+2025 Table 2:
  F300M/F335M/F360M   k = 2.07 +/- 0.30 (PDRs4All PAHFIT; F360M contaminated)
  F560W/F770W/F1000W  k = 4.33 +/- 0.35 (PDRs4All PAHFIT; F560W contaminated)
  F1000W/F1130W/F1500W k = 7.21 +/- 0.92 (PDRs4All Polyfit; F1500W contaminated)

Differences vs Tarantino+2025:
  - per-pixel ERR maps (propagated through PSF matching) instead of a
    single empirical RMS; the zero-point systematic ZPSYS of each band
    is added in quadrature to its ERR map before propagation.
  - inputs are the zero-point homogenized maps (products/matched/).

Inputs : products/matched/{filt}_matchedF2100W.fits (SCI, ERR)
Outputs: products/pah/{filt}_pah.fits  (PAH, PAH_ERR, CON, SLOPE)
         products/pah/pah_maps_summary.png
         products/pah/{filt}_pah_diagnostic.png

Usage: conda activate jwst && python 01_consub_kmethod.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.visualization import simple_norm

# ─── Configuration ────────────────────────────────────────
ZP_DIR  = os.path.expanduser('~/SMC_GO5952/products/matched')
OUT_DIR = os.path.expanduser('~/SMC_GO5952/products/pah')

# pivot wavelengths [um], JDox (only beta = (l2-l1)/(l3-l1) matters)
PIVOT = {'F300M': 2.996, 'F335M': 3.365, 'F360M': 3.621,
         'F560W': 5.635, 'F770W': 7.639, 'F1000W': 9.953,
         'F1130W': 11.309, 'F1500W': 15.064}

# trio: (f1_low, f2_mid, f3_up), contaminated flank, adopted k1 (Tarantino+2025 Table 2)
TRIOS = {
    '3.3':  dict(filts=('F300M', 'F335M', 'F360M'),  contam='up',  k=2.07, k_err=0.30),
    '7.7':  dict(filts=('F560W', 'F770W', 'F1000W'), contam='low', k=4.33, k_err=0.35),
    '11.3': dict(filts=('F1000W', 'F1130W', 'F1500W'), contam='up', k=7.21, k_err=0.92),
}

os.makedirs(OUT_DIR, exist_ok=True)


# ─── k-method equations (ported from Tarantino k_eq_with_err.py) ──
def get_pah_low(f1, f2, f3, l1, l2, l3, k, e1, e2, e3, k_err):
    """Contaminated flank = lower-wavelength filter (F560W/F770W/F1000W).
    f1 = fc1 + fp1 ; f2 = fc2 + fp2 ; f3 = fc3 ; fp2 = k*fp1."""
    beta = (l2 - l1) / (l3 - l1)
    fp1 = (f1 * (1 - beta) + f3 * beta - f2) / (1 - k - beta)
    fp2 = fp1 * k
    fc1 = f1 - fp1
    fc2 = ((f3 - fc1) / (l3 - l1)) * (l2 - l1) + fc1
    slope = (f3 - fc1) / (l3 - l1)

    # exact total derivatives of fp2 = k*fp1 (NOT k*sigma(fp1): the k
    # dependences of the prefactor and of fp1 partially cancel —
    # d(fp2)/dk = X(1-beta)/D^2, a factor (1-beta)/k smaller than the
    # k*d(fp1)/dk used in Tarantino's code, which is conservative)
    X = f1 * (1 - beta) + f3 * beta - f2
    D = 1 - k - beta
    t1 = (k * (1 - beta) / D) ** 2 * e1 ** 2
    t2 = (k / D) ** 2 * e2 ** 2
    t3 = (k * beta / D) ** 2 * e3 ** 2
    t4 = (X * (1 - beta) / D ** 2) ** 2 * k_err ** 2
    fp2_err = np.sqrt(t1 + t2 + t3 + t4)
    return dict(pah=fp2, con=fc2, slope=slope, pah_err=fp2_err)


def get_pah_up(f1, f2, f3, l1, l2, l3, k, e1, e2, e3, k_err):
    """Contaminated flank = upper-wavelength filter
    (F300M/F335M/F360M and F1000W/F1130W/F1500W).
    f1 = fc1 ; f2 = fc2 + fp2 ; f3 = fc3 + fp3 ; fp2 = k*fp3."""
    beta = (l2 - l1) / (l3 - l1)
    fp3 = (f1 * (1 - beta) + f3 * beta - f2) / (beta - k)
    fp2 = fp3 * k
    fc3 = f3 - fp3
    fc2 = ((fc3 - f1) / (l3 - l1)) * (l2 - l1) + f1
    slope = (fc3 - f1) / (l3 - l1)

    # exact total derivatives of fp2 = k*fp3 (see note in get_pah_low:
    # d(fp2)/dk = X*beta/D^2, a factor beta/k smaller than Tarantino's
    # conservative k*d(fp3)/dk)
    X = f1 * (1 - beta) + f3 * beta - f2
    D = beta - k
    t1 = (k * (1 - beta) / D) ** 2 * e1 ** 2
    t2 = (k / D) ** 2 * e2 ** 2
    t3 = (k * beta / D) ** 2 * e3 ** 2
    t4 = (X * beta / D ** 2) ** 2 * k_err ** 2
    fp2_err = np.sqrt(t1 + t2 + t3 + t4)
    return dict(pah=fp2, con=fc2, slope=slope, pah_err=fp2_err)


# ─── I/O helpers ──────────────────────────────────────────
def load_band(filt):
    path = os.path.join(ZP_DIR, f'{filt}_matchedF2100W.fits')
    with fits.open(path) as hdu:
        sci = hdu['SCI'].data
        err = hdu['ERR'].data
        hdr = hdu['SCI'].header
        zpsys = hdr['ZPSYS']
    # zero-point systematic added in quadrature to the per-pixel errors
    err_eff = np.sqrt(err ** 2 + zpsys ** 2)
    return sci, err_eff, hdr


# reference dark zone (for empirical noise check on the PAH maps)
ref = fits.getdata(os.path.join(ZP_DIR, 'zeropoint_refzones.fits')) == 1

# ─── Run the three trios ──────────────────────────────────
results = {}
for band, cfg in TRIOS.items():
    fl, fm, fu = cfg['filts']
    k, k_err = cfg['k'], cfg['k_err']
    f1, e1, hdr1 = load_band(fl)
    f2, e2, hdr2 = load_band(fm)
    f3, e3, _ = load_band(fu)

    func = get_pah_low if cfg['contam'] == 'low' else get_pah_up
    res = func(f1, f2, f3, PIVOT[fl], PIVOT[fm], PIVOT[fu],
               k, e1, e2, e3, k_err)
    results[band] = (fm, res)

    # save: PAH, PAH_ERR, CON, SLOPE
    hdr = hdr2.copy()
    for key, val, com in [
            ('PAHBAND', band, 'PAH feature [um]'),
            ('TRIO', f'{fl}/{fm}/{fu}', 'filter trio f1/f2/f3'),
            ('KCONTAM', k, 'PAH contamination constant (Tarantino+2025 k1)'),
            ('KERR', k_err, 'uncertainty on KCONTAM'),
            ('KCONTFLT', fu if cfg['contam'] == 'up' else fl,
             'PAH-contaminated flanking filter')]:
        hdr[key] = (val, com)
    hdus = fits.HDUList([fits.PrimaryHDU(header=fits.Header(
        [('COMMENT', 'k-method continuum subtraction, con_sub/01_consub_kmethod.py')]))])
    for name, arr in [('PAH', res['pah']), ('PAH_ERR', res['pah_err']),
                      ('CON', res['con']), ('SLOPE', res['slope'])]:
        h = hdr.copy()
        h['EXTNAME'] = name
        hdus.append(fits.ImageHDU(arr.astype(np.float32), header=h))
    out = os.path.join(OUT_DIR, f'{fm}_pah.fits')
    hdus.writeto(out, overwrite=True)

    # stats
    snr = res['pah'] / res['pah_err']
    n_val = np.isfinite(snr).sum()
    n3, n5 = (snr > 3).sum(), (snr > 5).sum()
    _, med_dark, rms_dark = sigma_clipped_stats(res['pah'][ref], sigma=3)
    med_err_dark = np.nanmedian(res['pah_err'][ref])
    print(f'PAH {band:4s} ({fm}, k={k}): SNR>3: {100 * n3 / n_val:.1f}% of pixels, '
          f'SNR>5: {100 * n5 / n_val:.1f}%')
    print(f'         dark zone: median={med_dark:+.4f}, empirical RMS={rms_dark:.4f}, '
          f'median propagated err={med_err_dark:.4f} MJy/sr '
          f'(ratio {rms_dark / med_err_dark:.2f})')

# ─── Figures ──────────────────────────────────────────────
# per-band diagnostics
for band, (fm, res) in results.items():
    fig, axes = plt.subplots(1, 4, figsize=(21, 5.5))
    norm = simple_norm(res['pah'], 'asinh', percent=99.5)
    ims = [(res['pah'], f'PAH {band} um (MJy/sr)', dict(norm=norm, cmap='magma')),
           (res['pah_err'], 'uncertainty', dict(vmin=0, vmax=np.nanpercentile(res['pah_err'], 98), cmap='magma')),
           (res['pah'] / res['pah_err'], 'SNR', dict(vmin=0, vmax=10, cmap='rainbow')),
           (res['con'], 'continuum in PAH filter', dict(norm=simple_norm(res['con'], 'asinh', percent=99.5), cmap='afmhot'))]
    for ax, (im, title, kw) in zip(axes, ims):
        m = ax.imshow(im, origin='lower', **kw)
        plt.colorbar(m, ax=ax, fraction=0.046)
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f'{fm} — k-method continuum subtraction (k={TRIOS[band]["k"]})', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'{fm}_pah_diagnostic.png'), dpi=110,
                bbox_inches='tight')
    plt.close()

# summary: the three PAH maps side by side
fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
for ax, (band, (fm, res)) in zip(axes, results.items()):
    norm = simple_norm(res['pah'], 'asinh', percent=99.5)
    m = ax.imshow(res['pah'], origin='lower', cmap='magma', norm=norm)
    plt.colorbar(m, ax=ax, fraction=0.046, label='MJy/sr')
    ax.set_title(f'PAH {band} $\\mu$m ({fm})', fontsize=13)
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'pah_maps_summary.png'), dpi=120,
            bbox_inches='tight')
plt.close()
print(f'\nWrote PAH maps and figures to {OUT_DIR}')
