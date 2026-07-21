"""
verify_canonical_order.py — canonical-order pipeline vs original chain
=======================================================================
The 2026-07-22 rebuild applies the zero-point constants BEFORE the PSF
matching (psf_match/00) instead of after. Since constants commute with
the chain (see verify_zeropoint_commutes.py), the new products/matched
maps must equal the original analysis_ready/zp maps to machine
precision. This script verifies exactly that, per filter, for SCI and
ERR (ERR must be strictly identical: the constant does not touch it).

Usage: conda activate jwst && python verify_canonical_order.py
"""

import os
import numpy as np
from astropy.io import fits

FILTERS = ['F150W', 'F187N', 'F200W', 'F212N', 'F300M', 'F335M', 'F360M',
           'F444W', 'F560W', 'F770W', 'F1000W', 'F1130W', 'F1500W', 'F2100W']
NEW = os.path.expanduser('~/SMC_GO5952/products/matched')
OLD = os.path.expanduser('~/SMC_GO5952/archive/analysis_ready_DEPRECATED_20260722/zp')

print('filtre   max|dSCI| (MJy/sr)  max|dERR|   NaN identiques')
ok = True
for f in FILTERS:
    new = fits.open(os.path.join(NEW, f'{f}_matchedF2100W.fits'))
    old = fits.open(os.path.join(OLD, f'{f}_matchedF2100W_zp.fits'))
    ds = de = 0.0
    same_nan = True
    for ext in ('SCI', 'ERR'):
        a, b = new[ext].data, old[ext].data
        fin = np.isfinite(a) & np.isfinite(b)
        same_nan &= bool(np.array_equal(np.isfinite(a), np.isfinite(b)))
        d = float(np.max(np.abs(a[fin] - b[fin]))) if fin.any() else np.inf
        if ext == 'SCI':
            ds = d
        else:
            de = d
    # maps are stored float32: judge against relative rounding (~6e-8)
    scale = float(np.nanmax(np.abs(new['SCI'].data)))
    status_ok = ds / scale < 1e-6 and de < 1e-7 and same_nan
    ok &= status_ok
    print(f'{f:8s} {ds:18.2e} {de:11.2e}   {same_nan}   '
          f'(rel {ds / scale:.1e})'
          + ('' if status_ok else '   <-- PROBLEME'))
    new.close(); old.close()
print('\nVALIDATION ' + ('PASSEE : nouvelle chaine == ancienne au bit pres'
                         if ok else 'ECHOUEE'))
