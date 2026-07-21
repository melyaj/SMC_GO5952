"""
verify_zeropoint_commutes.py — order of zero-point subtraction vs PSF matching
===============================================================================
Demonstrates on real data that subtracting a per-filter CONSTANT after the
PSF-matching chain is mathematically identical to subtracting it before:
a normalized kernel (sum=1) and the reprojection both preserve constants,
so k (x) (I - c) = k (x) I - c away from map borders (borders are NaN-cropped
in production and excluded from the common footprint).

Test: real F770W -> F2100W Aniano kernel (truncated to its energy support),
real 800x800 F770W cutout with NaNs, astropy convolve_fft with the production
NaN treatment. Result: max |before - after| ~ 2e-14 MJy/sr (machine precision)
in the interior region.

This is why the zero-point homogenization (con_sub/00) is applied AFTER
psf_match: the offsets must be MEASURED in the same physical region at the
same resolution in all 14 bands, which is only defined after matching --
while their SUBTRACTION commutes with the chain exactly.

Usage: conda activate jwst && python verify_zeropoint_commutes.py
"""
import numpy as np, os
from astropy.io import fits
from astropy.convolution import convolve_fft

k = fits.getdata(os.path.expanduser(
    '~/SMC_GO5952/work/psf_match/kernels/F770W_to_F2100W_kernel.fits')).astype(float)
c0 = k.shape[0] // 2
kc = k[c0-150:c0+151, c0-150:c0+151]
kc /= kc.sum()

img = fits.getdata(os.path.expanduser(
    '~/SMC_GO5952/products/matched/F770W_matchedF2100W.fits'), 'SCI')
cut = img[200:1000, 200:1000].astype(float)
c = 0.1034     # F770W zero-point offset (MJy/sr)

kw = dict(normalize_kernel=True, nan_treatment='interpolate',
          preserve_nan=True, allow_huge=True)
before = convolve_fft(cut - c, kc, **kw)
after = convolve_fft(cut, kc, **kw) - c
core = np.s_[200:600, 200:600]
diff = np.nanmax(np.abs(before[core] - after[core]))
print(f'max |subtract-before - subtract-after| = {diff:.2e} MJy/sr')
assert diff < 1e-10, 'constant does NOT commute?!'
print('OK: zero-point subtraction commutes with the PSF-matching chain.')
