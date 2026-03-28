"""
skysub_utils.py — 2D polynomial sky subtraction for JWST i2d mosaics
=====================================================================
SMC GO-5952 custom data reduction.

Removes residual large-scale background gradients from Stage 3 mosaics
by fitting a low-order 2D polynomial to the background (masking sources).

Following the approach suggested by C. Clark (Clark+2021, ApJ, 921, 35),
simplified from the full HI-to-dust spatially-varying ratio to a simple polynomial.

Usage:
    sys.path.insert(0, "..")
    from skysub_utils import *

Functions:
    make_source_mask        — Build a combined source mask (segmentation + sigma clip + dilation)
    fit_2d_polynomial       — Fit a 2D polynomial to unmasked pixels
    subtract_sky_polynomial — Full pipeline: mask → fit → subtract → save

References:
    - Clark, C. J. R., et al. 2021, ApJ, 921, 35
"""

import os
import warnings
import numpy as np

from astropy.io import fits
from astropy.stats import sigma_clip as astropy_sigma_clip
from astropy.stats import sigma_clipped_stats
from scipy.ndimage import binary_dilation


# ═══════════════════════════════════════════════════════════
# SOURCE MASKING
# ═══════════════════════════════════════════════════════════

def make_source_mask(data, segm_file=None, nsigma=3.0, grow_npix=5):
    """
    Build a combined source mask for background fitting.

    Combines up to three masking layers:
    1. Segmentation map (if provided) — masks all detected sources
    2. Sigma clipping — catches faint sources not in the segmentation map
    3. Dilation — grows the mask to exclude source wings

    Parameters
    ----------
    data : 2D ndarray
        Image data.
    segm_file : str or None
        Path to a segmentation map FITS file. Pixels > 0 are masked.
    nsigma : float
        Sigma threshold for source masking.
    grow_npix : int
        Number of pixels to grow the mask by.

    Returns
    -------
    mask : 2D bool array
        True where pixels are masked.
    mean_bg : float
        Sigma-clipped background mean.
    std_bg : float
        Sigma-clipped background std.
    """
    # Start with NaN/zero mask
    mask = np.isnan(data) | (data == 0)

    # Segmentation map
    if segm_file is not None and os.path.exists(segm_file):
        segm_data = fits.getdata(segm_file)
        segm_mask = segm_data > 0
        n_sources = len(np.unique(segm_data[segm_data > 0]))
        print(f"  Segmentation map: {n_sources} sources, "
              f"{np.sum(segm_mask)} pixels ({100*np.sum(segm_mask)/segm_mask.size:.1f}%)")
        mask |= segm_mask

    # Sigma clipping
    valid = data[~mask]
    clipped = astropy_sigma_clip(valid, sigma_lower=5, sigma_upper=nsigma,
                                  maxiters=10, masked=True)
    mean_bg = np.mean(clipped.compressed())
    std_bg = np.std(clipped.compressed())
    print(f"  Background stats: mean = {mean_bg:.4f}, std = {std_bg:.4f}")

    source_mask = data > (mean_bg + nsigma * std_bg)
    mask |= source_mask
    print(f"  Sigma clip (>{nsigma}σ): {np.sum(source_mask)} additional pixels")

    # Grow mask
    if grow_npix > 0:
        struct = np.ones((2*grow_npix+1, 2*grow_npix+1), dtype=bool)
        n_before = np.sum(mask)
        mask = binary_dilation(mask, structure=struct)
        print(f"  Grew mask by {grow_npix} px: {np.sum(mask) - n_before} additional pixels")

    print(f"  Final mask: {np.sum(mask)}/{data.size} pixels "
          f"({100*np.sum(mask)/data.size:.1f}%)")

    return mask, mean_bg, std_bg


# ═══════════════════════════════════════════════════════════
# POLYNOMIAL FITTING
# ═══════════════════════════════════════════════════════════

def fit_2d_polynomial(data, mask, degree=2):
    """
    Fit a 2D polynomial to unmasked pixels.

    Uses normalized coordinates [-1, 1] for numerical stability.

    Parameters
    ----------
    data : 2D ndarray
        Image data.
    mask : 2D bool array
        True where pixels are masked (excluded from fit).
    degree : int
        Polynomial degree (2 = quadratic, 3 = cubic).

    Returns
    -------
    bg_model : 2D ndarray
        Fitted polynomial background surface.
    coeffs : array
        Fitted polynomial coefficients.
    """
    ny, nx = data.shape

    # Normalized coordinate grids
    yy, xx = np.mgrid[0:ny, 0:nx]
    xx_norm = 2.0 * xx / (nx - 1) - 1.0
    yy_norm = 2.0 * yy / (ny - 1) - 1.0

    # Build polynomial terms
    terms = []
    term_labels = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            terms.append((xx_norm ** j) * (yy_norm ** i))
            term_labels.append(f"x^{j} * y^{i}")

    n_terms = len(terms)
    print(f"  Polynomial degree {degree}: {n_terms} terms")

    # Unmasked pixels
    good = ~mask
    n_good = np.sum(good)
    print(f"  Fitting to {n_good} unmasked pixels")

    if n_good < n_terms:
        raise ValueError(f"Not enough unmasked pixels ({n_good}) for "
                         f"degree {degree} ({n_terms} terms)")

    # Solve least squares
    A = np.column_stack([t[good] for t in terms])
    b = data[good]
    coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    print(f"  Coefficients:")
    for lbl, c in zip(term_labels, coeffs):
        print(f"    {lbl}: {c:.6e}")

    # Evaluate on full grid
    A_full = np.column_stack([t.ravel() for t in terms])
    bg_model = (A_full @ coeffs).reshape(ny, nx)

    # Residual stats
    res = data[good] - bg_model[good]
    print(f"  Residuals (unmasked): mean = {np.nanmean(res):.6e}, "
          f"std = {np.nanstd(res):.6e}")

    return bg_model, coeffs


# ═══════════════════════════════════════════════════════════
# FULL PIPELINE
# ═══════════════════════════════════════════════════════════

def subtract_sky_polynomial(input_file, output_file=None, segm_file=None,
                            poly_degree=2, nsigma=3.0, grow_npix=5):
    """
    Full 2D polynomial sky subtraction pipeline.

    Loads an i2d mosaic, masks sources, fits a polynomial to the
    background, subtracts it, and saves the result with the model
    and mask as extra FITS extensions.

    Parameters
    ----------
    input_file : str
        Path to input *_i2d.fits file.
    output_file : str or None
        Output path. If None, auto-generated as *_i2d_skysub.fits.
    segm_file : str or None
        Path to segmentation map for source masking.
    poly_degree : int
        Polynomial degree (2 or 3).
    nsigma : float
        Sigma clipping threshold for source masking.
    grow_npix : int
        Pixels to grow the source mask by.

    Returns
    -------
    output_file : str
        Path to the saved output file.
    data_sub : 2D ndarray
        Background-subtracted data.
    bg_model : 2D ndarray
        Fitted polynomial surface.
    mask : 2D bool array
        Source mask used for fitting.
    """
    print(f"Input: {input_file}")

    # Load
    hdul = fits.open(input_file)
    if 'SCI' in [h.name for h in hdul]:
        data = hdul['SCI'].data.copy()
        header = hdul['SCI'].header.copy()
        sci_idx = [h.name for h in hdul].index('SCI')
    elif len(hdul) > 1:
        data = hdul[1].data.copy()
        header = hdul[1].header.copy()
        sci_idx = 1
    else:
        data = hdul[0].data.copy()
        header = hdul[0].header.copy()
        sci_idx = 0

    print(f"Image shape: {data.shape}")

    # Mask sources
    print("--- Source Masking ---")
    mask, mean_bg, std_bg = make_source_mask(data, segm_file=segm_file,
                                              nsigma=nsigma, grow_npix=grow_npix)

    # Fit polynomial
    print(f"--- Polynomial Fit (degree={poly_degree}) ---")
    bg_model, coeffs = fit_2d_polynomial(data, mask, degree=poly_degree)

    # Subtract
    data_sub = data - bg_model
    data_sub[np.isnan(data)] = np.nan

    # Stats
    good_sub = ~np.isnan(data_sub) & ~mask
    print(f"--- Result ---")
    print(f"  Background: mean = {np.nanmean(data_sub[good_sub]):.6e}, "
          f"std = {np.nanstd(data_sub[good_sub]):.6e}")

    # Save
    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_skysub{ext}"

    hdul_out = fits.open(input_file)
    hdul_out[sci_idx].data = data_sub
    hdul_out[sci_idx].header['HISTORY'] = f'Sky subtracted: 2D polynomial degree {poly_degree}'
    hdul_out[sci_idx].header['HISTORY'] = f'Sigma clip = {nsigma}, grow = {grow_npix} px'
    hdul_out[sci_idx].header['SKYPOLY'] = (poly_degree, 'Sky polynomial degree')

    if segm_file and os.path.exists(segm_file):
        hdul_out[sci_idx].header['HISTORY'] = f'Segmentation mask: {os.path.basename(segm_file)}'

    # Background model extension
    bg_hdu = fits.ImageHDU(data=bg_model, header=header, name='SKYMODEL')
    bg_hdu.header['COMMENT'] = 'Fitted 2D polynomial sky model'
    hdul_out.append(bg_hdu)

    # Mask extension
    mask_hdu = fits.ImageHDU(data=mask.astype(np.uint8), header=header, name='SKYMASK')
    mask_hdu.header['COMMENT'] = 'Source mask used for sky fit (1=masked)'
    hdul_out.append(mask_hdu)

    hdul_out.writeto(output_file, overwrite=True)
    hdul_out.close()
    hdul.close()

    print(f"Saved: {output_file}")
    return output_file, data_sub, bg_model, mask
