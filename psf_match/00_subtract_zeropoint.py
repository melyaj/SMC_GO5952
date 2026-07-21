"""
00_subtract_zeropoint.py — apply per-filter zero-point constants at mosaic level
=================================================================================
SMC GO-5952 — psf_match module, step 00.

Canonical-order pipeline: ALL background removal happens BEFORE PSF
matching. This step subtracts the per-filter zero-point constant
(pedestal) from the stage-3 mosaics, so that the PSF-matched products
are directly science-ready.

The constants come from `zeropoint_offsets.ecsv` (versioned alongside
this script): sigma-clipped medians measured in the common dark
reference cavity ON THE MATCHED MAPS — the only place where the same
physical region is sampled identically in all 14 bands. Measuring
there and subtracting here is exact: a normalized kernel and the
reprojection preserve constants to machine precision (see
con_sub/tests/verify_zeropoint_commutes.py).

Inputs : stage3 mosaics (MIRI: *_i2d_skysub.fits, NIRCam: *_i2d.fits)
Outputs: same, with suffix _zp.fits, SCI = SCI - offset,
         header ZPOFF/ZPSYS on SCI and ERR (ERR unchanged).

Usage: conda activate jwst && python 00_subtract_zeropoint.py
"""

from astropy.io import fits
from astropy.table import Table

from config import ALL_FILTERS, i2d_path_raw, i2d_path, ZP_TABLE

offsets = Table.read(ZP_TABLE, format='ascii.ecsv')
off = {r['filter']: (float(r['offset']), float(r['sys_scatter']))
       for r in offsets}

for filt in ALL_FILTERS:
    src, dst = i2d_path_raw(filt), i2d_path(filt)
    zpoff, zpsys = off[filt]
    with fits.open(src) as hdul:
        hdul['SCI'].data = hdul['SCI'].data - zpoff
        for ext in ('SCI', 'ERR'):
            hdul[ext].header['ZPOFF'] = (
                zpoff, '[MJy/sr] zero-point offset subtracted from SCI')
            hdul[ext].header['ZPSYS'] = (
                zpsys, '[MJy/sr] systematic zero uncertainty (zone scatter)')
        hdul[0].header['HISTORY'] = (
            'Zero-point constant subtracted at mosaic level '
            '(psf_match/00_subtract_zeropoint.py)')
        hdul.writeto(dst, overwrite=True)
    print(f'{filt:7s} {zpoff:+8.4f} MJy/sr -> {dst.name}')
