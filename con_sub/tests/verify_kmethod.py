"""
verify_kmethod.py — independent verification of the k-method implementation
============================================================================
SMC GO-5952 — con_sub module, tests.

Four checks, all independent of the production code path where possible:

A. Algebraic identity: our get_pah_up/get_pah_low reproduce Eq. 5 of
   Tarantino+2025 (fp2 = k/(k-beta) [f2 - (1-beta) f1 - beta f3]) for
   random inputs, in both contamination configurations.
B. Exact recovery: build synthetic bands from a KNOWN linear continuum
   + KNOWN PAH fluxes satisfying the contamination assumption
   (fp_contam = fp2/k) and verify the method recovers the input PAH
   flux to machine precision; also f2 = con + pah must hold.
C. Monte-Carlo error propagation: compare the analytic pah_err to the
   empirical std over 200k noisy realizations (Gaussian noise on
   f1/f2/f3 and on k), for the three real trios at representative
   surface brightnesses.
D. Real-config sanity: beta per trio from measured pivot wavelengths
   (recomputed from the stpsf throughput curves, photon-weighted),
   distance of the denominators (k - beta etc.) from zero, and
   sensitivity d(fp2)/fp2 per 10% change of k.

Usage: conda activate jwst && python verify_kmethod.py
"""

import os
import sys
import numpy as np
from astropy.table import Table

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.expanduser('~/SMC_GO5952/pipeline/con_sub'))
import importlib.util
spec = importlib.util.spec_from_file_location(
    'consub', os.path.expanduser(
        '~/SMC_GO5952/pipeline/con_sub/01_consub_kmethod.py'))
# import only the two functions without running the pipeline body
import types, re
src = open(os.path.expanduser(
    '~/SMC_GO5952/pipeline/con_sub/01_consub_kmethod.py')).read()
func_src = src[src.index('def get_pah_low'):src.index('# ─── I/O helpers')]
mod = types.ModuleType('kfuncs')
mod.np = np
exec(func_src, mod.__dict__)
get_pah_low, get_pah_up = mod.get_pah_low, mod.get_pah_up

rng = np.random.default_rng(42)
STPSF = os.path.expanduser('~/data/stpsf-data')

print('=' * 66)
print('A. ALGEBRAIC IDENTITY vs Tarantino+2025 Eq. 5')
print('=' * 66)
ok = True
for case, func in [('up', get_pah_up), ('low', get_pah_low)]:
    f1, f2, f3 = rng.uniform(0.1, 20, (3, 100000))
    l1, l2, l3 = 5.635, 7.639, 9.953
    k = rng.uniform(1.5, 10, 100000)
    beta = (l2 - l1) / (l3 - l1)
    eq5 = k / (k - beta) * (f2 - (1 - beta) * f1 - beta * f3)
    res = func(f1, f2, f3, l1, l2, l3, k, 0, 0, 0, 0)
    if case == 'up':
        # up-config: same functional form, Eq.5 as written in the paper
        diff = np.max(np.abs(res['pah'] - eq5) / np.abs(eq5))
    else:
        # low-config: contaminated flank is f1 -> denominator (k-1+beta)
        eq5_low = k / (k - 1 + beta) * (f2 - (1 - beta) * f1 - beta * f3)
        diff = np.max(np.abs(res['pah'] - eq5_low) / np.abs(eq5_low))
    print(f'  {case:3s}: max relative difference = {diff:.2e}  '
          f'{"OK" if diff < 1e-12 else "FAIL"}')
    ok &= diff < 1e-12

print()
print('=' * 66)
print('B. EXACT RECOVERY of known PAH flux (assumptions satisfied)')
print('=' * 66)
for case, func, trio in [
        ('up  (3.3)', get_pah_up, (2.996, 3.365, 3.621)),
        ('low (7.7)', get_pah_low, (5.635, 7.639, 9.953)),
        ('up  (11.3)', get_pah_up, (9.953, 11.309, 15.064))]:
    l1, l2, l3 = trio
    a, b = rng.uniform(0.5, 2), rng.uniform(-0.1, 0.3)   # linear continuum
    fc = {l: a + b * (l - l1) for l in trio}
    fp2_true = rng.uniform(0.5, 5)
    k = 3.7
    if 'up' in case:     # contaminated = f3
        f1 = fc[l1]
        f3 = fc[l3] + fp2_true / k
        f2 = fc[l2] + fp2_true
    else:                # contaminated = f1
        f1 = fc[l1] + fp2_true / k
        f3 = fc[l3]
        f2 = fc[l2] + fp2_true
    res = func(np.array([f1]), np.array([f2]), np.array([f3]),
               l1, l2, l3, k, 0, 0, 0, 0)
    err_pah = abs(res['pah'][0] - fp2_true) / fp2_true
    err_sum = abs(res['pah'][0] + res['con'][0] - f2) / f2
    print(f'  {case}: input fp2={fp2_true:.4f}, recovered='
          f'{res["pah"][0]:.4f} (rel err {err_pah:.1e}); '
          f'pah+con=f2 to {err_sum:.1e}  '
          f'{"OK" if err_pah < 1e-12 and err_sum < 1e-12 else "FAIL"}')
    ok &= err_pah < 1e-12 and err_sum < 1e-12

print()
print('=' * 66)
print('C. MONTE-CARLO check of the analytic error propagation')
print('=' * 66)
N = 200000
configs = [
    ('3.3 faint',  get_pah_up,  (2.996, 3.365, 3.621), 2.07, 0.30,
     (0.5, 0.55, 0.52), (0.019, 0.014, 0.018)),
    ('3.3 bright', get_pah_up,  (2.996, 3.365, 3.621), 2.07, 0.30,
     (2.0, 2.8, 2.4),   (0.019, 0.014, 0.018)),
    ('7.7',        get_pah_low, (5.635, 7.639, 9.953), 4.33, 0.35,
     (1.0, 4.0, 2.5),   (0.02, 0.02, 0.03)),
    ('11.3',       get_pah_up,  (9.953, 11.309, 15.064), 7.21, 0.92,
     (2.5, 6.0, 4.0),   (0.03, 0.05, 0.09)),
]
for name, func, (l1, l2, l3), k0, kerr, (m1, m2, m3), (s1, s2, s3) in configs:
    f1 = rng.normal(m1, s1, N)
    f2 = rng.normal(m2, s2, N)
    f3 = rng.normal(m3, s3, N)
    kk = rng.normal(k0, kerr, N)
    mc = func(f1, f2, f3, l1, l2, l3, kk, 0, 0, 0, 0)['pah']
    ana = func(np.array([m1]), np.array([m2]), np.array([m3]),
               l1, l2, l3, k0, s1, s2, s3, kerr)
    ratio = np.std(mc) / ana['pah_err'][0]
    print(f'  {name:10s}: MC std = {np.std(mc):.4f}, analytic = '
          f'{ana["pah_err"][0]:.4f}  (MC/analytic = {ratio:.3f})  '
          f'{"OK" if 0.9 < ratio < 1.15 else "CHECK"}')

print()
print('=' * 66)
print('D. REAL CONFIG: pivots from throughputs, beta, denominators,')
print('   and sensitivity to k')
print('=' * 66)
JDOX = {'F300M': 2.996, 'F335M': 3.365, 'F360M': 3.621,
        'F560W': 5.635, 'F770W': 7.639, 'F1000W': 9.953,
        'F1130W': 11.309, 'F1500W': 15.064}
INST = {'F300M': 'NIRCam', 'F335M': 'NIRCam', 'F360M': 'NIRCam',
        'F560W': 'MIRI', 'F770W': 'MIRI', 'F1000W': 'MIRI',
        'F1130W': 'MIRI', 'F1500W': 'MIRI'}
piv = {}
for f, inst in INST.items():
    t = Table.read(os.path.join(STPSF, inst, 'filters',
                                f'{f}_throughput.fits'))
    w = t['WAVELENGTH'] / 1e4
    T = t['THROUGHPUT']
    piv[f] = float(np.sqrt(np.trapz(T * w, w) / np.trapz(T / w, w)))
    dev = 1e4 * (piv[f] - JDOX[f]) / JDOX[f]
    print(f'  {f:7s} pivot: measured {piv[f]:.4f} um vs JDox '
          f'{JDOX[f]:.4f} ({dev:+.1f} x1e-4)')

print()
for name, (a, b, c), k0, kerr, case in [
        ('3.3',  ('F300M', 'F335M', 'F360M'), 2.07, 0.30, 'up'),
        ('7.7',  ('F560W', 'F770W', 'F1000W'), 4.33, 0.35, 'low'),
        ('11.3', ('F1000W', 'F1130W', 'F1500W'), 7.21, 0.92, 'up')]:
    l1, l2, l3 = piv[a], piv[b], piv[c]
    beta = (l2 - l1) / (l3 - l1)
    den = (beta - k0) if case == 'up' else (1 - k0 - beta)
    func = get_pah_up if case == 'up' else get_pah_low
    # sensitivity: rerun trio B-style synthetic point with k +/- 10%
    fc = {l: 1.0 + 0.1 * (l - l1) for l in (l1, l2, l3)}
    fp2 = 1.0
    if case == 'up':
        f1, f2, f3 = fc[l1], fc[l2] + fp2, fc[l3] + fp2 / k0
    else:
        f1, f2, f3 = fc[l1] + fp2 / k0, fc[l2] + fp2, fc[l3]
    outs = [func(np.array([f1]), np.array([f2]), np.array([f3]),
                 l1, l2, l3, kx, 0, 0, 0, 0)['pah'][0]
            for kx in (0.9 * k0, k0, 1.1 * k0)]
    sens = 100 * (outs[2] - outs[0]) / (0.2 * outs[1])
    print(f'  trio {name:4s}: beta = {beta:.3f}, |denominator| = '
          f'{abs(den):.2f} (safe if >>0), d(fp2)/fp2 per +10% k = '
          f'{sens / 10:+.2f}%')

print()
print('ALL ALGEBRA CHECKS PASSED' if ok else 'SOME CHECKS FAILED')
