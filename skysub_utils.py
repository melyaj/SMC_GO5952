"""
skysub_utils.py — 2D polynomial sky subtraction for JWST i2d mosaics
=====================================================================
SMC GO-5952 custom data reduction.
Elyajouri Meriem (STScI) 30/03/2026

Removes residual large-scale background gradients from Stage 3 mosaics
by fitting a low-order 2D polynomial to the background (masking sources).

Usage:
    sys.path.insert(0, "..")
    from skysub_utils import *

Functions:
    I/O:
        load_i2d                   — Load an i2d FITS mosaic (zeros → NaN)
        save_skysub                — Save subtracted mosaic with model and mask extensions

    Processing:
        make_source_mask           — Build source mask (segmentation + sigma clip + dilation + edge crop)
        fit_2d_polynomial          — Fit a 2D polynomial to unmasked pixels
        subtract_sky_polynomial    — Full pipeline: mask → fit → subtract → save
        compute_uncertainties      — Pixel noise, background level, polynomial bias

    Diagnostics/Figs:
        plot_mask_overlay          — Image + cyan mask contours (2 panels)
                                     → saves {FILT}_mask_overlay.png
        plot_bg_model              — Polynomial model + histogram + 1D cuts (3 panels)
                                     → saves {FILT}_bg_model.png
        plot_before_after          — Before/after sky subtraction (2 panels, stats overlay)
                                     → saves {FILT}_skysub_comparison.png
        plot_background_histogram  — Background pixel distribution vs Gaussian (1 panel)
                                     → saves {FILT}_background_histogram.png
        print_summary              — Print sky subtraction summary
"""

import os
import warnings
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.stats import sigma_clip as astropy_sigma_clip
from astropy.stats import sigma_clipped_stats
from astropy.visualization import simple_norm
from scipy.ndimage import binary_dilation


def load_i2d(input_file):
    """Load an i2d FITS mosaic, zeros → NaN."""
    hdul = fits.open(input_file)
    if 'SCI' in [h.name for h in hdul]:
        data = hdul['SCI'].data.copy()
        header = hdul['SCI'].header.copy()
    else:
        data = hdul[1].data.copy()
        header = hdul[1].header.copy()
    data[data == 0] = np.nan
    hdul.close()
    return data, header

# ═══════════════════════════════════════════════════════════
# SOURCE MASKING
# ═══════════════════════════════════════════════════════════

def make_source_mask(data, segm_file=None, nsigma=3.0, grow_npix=5, edge_crop=0):
    """
    Build a combined source mask for background fitting.

    Combines up to four masking layers:
    1. Segmentation map (if provided) — masks all detected sources,
       see https://jwst-pipeline.readthedocs.io/en/latest/jwst/source_catalog/main.html
    2. Sigma clipping — catches extended diffuse emission not in the segmentation map
    3. Dilation — grows the source mask to catch faint wings around bright sources
    4. Edge cropping — masks the noisy mosaic borders (irregular shape from dithering)

    Parameters
    ----------
    data : 2D ndarray
    segm_file : str or None
        Path to a segmentation map FITS file. Pixels > 0 are masked.
    nsigma : float
        Sigma threshold for source masking.
    grow_npix : int
        Number of pixels to grow the mask by (binary dilation).
    edge_crop : int
        Number of pixels to mask at mosaic edges (binary erosion).

    Returns
    -------
    mask : 2D bool array
        True where pixels are masked.
    mean_bg : float
        Sigma-clipped background mean.
    std_bg : float
        Sigma-clipped background std.
    """
    mask = np.isnan(data) | (data == 0)

    if segm_file is not None and os.path.exists(segm_file):
        segm_data = fits.getdata(segm_file)
        segm_mask = segm_data > 0
        n_sources = len(np.unique(segm_data[segm_data > 0]))
        print(f"  Segmentation map: {n_sources} sources, "
              f"{np.sum(segm_mask)} pixels ({100*np.sum(segm_mask)/segm_mask.size:.1f}%)")
        mask |= segm_mask

    valid = data[~mask]
    clipped = astropy_sigma_clip(valid, sigma_lower=5, sigma_upper=nsigma,
                                  maxiters=10, masked=True)
    mean_bg = np.mean(clipped.compressed())
    std_bg = np.std(clipped.compressed())
    print(f"  Background stats: mean = {mean_bg:.4f}, std = {std_bg:.4f}")

    source_mask = data > (mean_bg + nsigma * std_bg)
    mask |= source_mask
    print(f"  Sigma clip (>{nsigma}σ): {np.sum(source_mask)} additional pixels")

    if grow_npix > 0:
        struct = np.ones((2*grow_npix+1, 2*grow_npix+1), dtype=bool)
        n_before = np.sum(mask)
        mask = binary_dilation(mask, structure=struct)
        print(f"  Grew mask by {grow_npix} px: {np.sum(mask) - n_before} additional pixels")

    # Mask mosaic edges — irregular borders from dithering.
    # binary_erosion shrinks any arbitrary shape inward by edge_crop pixels.
    if edge_crop > 0:
        from scipy.ndimage import binary_erosion
        valid = ~np.isnan(data) & (data != 0)
        struct = np.ones((2*edge_crop+1, 2*edge_crop+1), dtype=bool)
        valid_eroded = binary_erosion(valid, structure=struct)
        edge_mask = valid & ~valid_eroded
        mask |= edge_mask
        print(f"  Edge crop ({edge_crop} px): {np.sum(edge_mask)} additional pixels masked")

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
    Without this, x² and y² terms reach ~10⁶, causing precision loss
    in the least-squares fit. The result is mathematically identical.

    Parameters
    ----------
    data : 2D ndarray
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
    yy, xx = np.mgrid[0:ny, 0:nx] #Creates two grids: xx holds the column number of every pixel, yy the row number. 
    xx_norm = 2.0 * xx / (nx - 1) - 1.0 # rescales coordinates from [0, 1800] to [-1, +1] 
    yy_norm = 2.0 * yy / (ny - 1) - 1.0

    terms = []
    term_labels = []
    for i in range(degree + 1): # Builds the 6 polynomial terms for degree 2. The loop generates: 1, x, x², y, xy, y². Each term is a full 2D image, for example the xy term is a grid where every pixel holds the product of its x and y position.
        for j in range(degree + 1 - i):
            terms.append((xx_norm ** j) * (yy_norm ** i))
            term_labels.append(f"x^{j} * y^{i}")

    n_terms = len(terms)
    print(f"  Polynomial degree {degree}: {n_terms} terms")

    good = ~mask # the background pixels (not masked). These are the only pixels we fit to.
    n_good = np.sum(good)
    print(f"  Fitting to {n_good} unmasked pixels")

    if n_good < n_terms:
        raise ValueError(f"Not enough unmasked pixels ({n_good}) for "
                         f"degree {degree} ({n_terms} terms)")

    A = np.column_stack([t[good] for t in terms]) # The design matrix of the linear system. Each row = 1 bckground pixel. Each column = 1 polynomial term. With 500,000 background pixels and 6 terms, A is 500000 × 6.
    b = data[good] # The measurement vector, the brightness value of each background pixel.
    coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None) # The core step: solves A × coeffs ≈ b in the least-squares sense. Finds the 6 coefficients (a, b, c, d, e, f) that minimize the total error. Same idea as fitting a line through a scatter plot, but in 6 dimensions.

    print(f"  Coefficients:")
    for lbl, c in zip(term_labels, coeffs):
        print(f"    {lbl}: {c:.6e}")

    A_full = np.column_stack([t.ravel() for t in terms]) # Now that we have the coeffs, evaluate the polynomial on every pixel (not just the background ones). A_full @ coeffs computes a + bx + cy + dx² + exy + fy² at each pixel. 
    bg_model = (A_full @ coeffs).reshape(ny, nx) # Reshape back to 2D -> that's our background model.

    res = data[good] - bg_model[good] # The residuals: what's left after subtracting the model, measured only on background pixels. If the fit is good, the mean should be ~0.
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
    Loads an i2d mosaic, masks sources, fits a polynomial,
    subtracts it, and saves with model and mask as extra extensions.
    """
    print(f"Input: {input_file}")

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

    print("--- Source Masking ---")
    mask, mean_bg, std_bg = make_source_mask(data, segm_file=segm_file,
                                              nsigma=nsigma, grow_npix=grow_npix)

    print(f"--- Polynomial Fit (degree={poly_degree}) ---")
    bg_model, coeffs = fit_2d_polynomial(data, mask, degree=poly_degree)

    data_sub = data - bg_model
    data_sub[np.isnan(data)] = np.nan

    good_sub = ~np.isnan(data_sub) & ~mask
    print(f"--- Result ---")
    print(f"  Background: mean = {np.nanmean(data_sub[good_sub]):.6e}, "
          f"std = {np.nanstd(data_sub[good_sub]):.6e}")

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

    bg_hdu = fits.ImageHDU(data=bg_model, header=header, name='SKYMODEL')
    bg_hdu.header['COMMENT'] = 'Fitted 2D polynomial sky model'
    hdul_out.append(bg_hdu)

    mask_hdu = fits.ImageHDU(data=mask.astype(np.uint8), header=header, name='SKYMASK')
    mask_hdu.header['COMMENT'] = 'Source mask used for sky fit (1=masked)'
    hdul_out.append(mask_hdu)

    hdul_out.writeto(output_file, overwrite=True)
    hdul_out.close()
    hdul.close()

    print(f"Saved: {output_file}")
    return output_file, data_sub, bg_model, mask


# ═══════════════════════════════════════════════════════════
# DIAGNOSTICS
# ═══════════════════════════════════════════════════════════

def plot_mask_overlay(data, mask, filt, cmap='afmhot', fig_dir='.', dpi=200):
    """Show the image with mask contours overlaid."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10), facecolor='none')

    norm = simple_norm(data, 'sqrt', percent=99)
    ax1.imshow(data, cmap=cmap, origin='lower', norm=norm)
    ax1.set_title('Original', fontsize=18, pad=10)
    ax1.axis('off')

    ax2.imshow(data, cmap=cmap, origin='lower', norm=norm)
    ax2.contour(mask.astype(float), levels=[0.5], colors='cyan', linewidths=0.5)
    ax2.set_title('Mask overlay (cyan = masked)', fontsize=18,  pad=10)
    ax2.axis('off')

    plt.tight_layout()
    fig.savefig(f'{fig_dir}/{filt}_mask_overlay.png',
                dpi=dpi, bbox_inches='tight', facecolor='none', transparent=True)
    plt.show()


def plot_bg_model(data, bg_model, filt, fig_dir='.', dpi=200):
    """
    Inspect the polynomial background model: image + 1D cuts through the center.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor='none')

    # Model image
    im = axes[0].imshow(bg_model, origin='lower', cmap='RdBu_r')
    axes[0].set_title('Polynomial background model', fontsize=14)
    axes[0].axis('off')
    plt.colorbar(im, ax=axes[0], shrink=0.8, label='MJy/sr')

    # Center cuts: one horizontal, one vertical
    ny, nx = bg_model.shape
    axes[1].plot(bg_model[ny//2, :], color='steelblue', lw=1.5,
                 label=f'Horizontal (row {ny//2})')
    axes[1].plot(bg_model[:, nx//2], color='darkorange', lw=1.5,
                 label=f'Vertical (col {nx//2})')
    axes[1].set_xlabel('Pixel')
    axes[1].set_ylabel('Model value (MJy/sr)')
    axes[1].set_title('1D cuts through center', fontsize=14)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(f'{fig_dir}/{filt}_bg_model.png',
                dpi=dpi, bbox_inches='tight', facecolor='none', transparent=True)
    plt.show()


def plot_before_after(data, data_sub, filt, poly_degree=2,
                      cmap='afmhot', fig_dir='.', dpi=200):
    """Before/after sky subtraction comparison with stats overlay."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10), facecolor='none')

    # Before
    d_before = data.copy().astype(float)
    d_before[d_before == 0] = np.nan
    norm_before = simple_norm(d_before, 'sqrt', percent=99)
    im1 = ax1.imshow(d_before, cmap=cmap, origin='lower', norm=norm_before)
    ax1.set_title('Before sky subtraction', fontsize=20,  pad=10)
    ax1.axis('off')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _, med_b, std_b = sigma_clipped_stats(d_before, sigma=3)
    neg_b = 100 * np.nansum(d_before < 0) / np.sum(np.isfinite(d_before))
    #ax1.text(0.03, 0.97,
    #         f'median = {med_b:.3f}\n$\\sigma$ = {std_b:.3f} MJy/sr',
    #         transform=ax1.transAxes, fontsize=14, va='top',
    #         color='white', family='monospace',
    #         bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.75))
    cb1 = plt.colorbar(im1, ax=ax1, orientation='horizontal',
                       fraction=0.046, pad=0.02, shrink=0.85)
    cb1.set_label('MJy/sr', fontsize=12)

    # After
    d_after = data_sub.copy().astype(float)
    d_after[d_after == 0] = np.nan
    norm_after = simple_norm(d_after, 'sqrt', percent=99)
    im2 = ax2.imshow(d_after, cmap=cmap, origin='lower', norm=norm_after)
    ax2.set_title('After sky subtraction', fontsize=20, pad=10)
    ax2.axis('off')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _, med_a, std_a = sigma_clipped_stats(d_after, sigma=3)
    #ax2.text(0.03, 0.97,
    #         f'median = {med_a:.3f}\n$\\sigma$ = {std_a:.3f} MJy/sr',
    #         transform=ax2.transAxes, fontsize=14, va='top',
    #         color='white', family='monospace',
    #         bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.75))
    cb2 = plt.colorbar(im2, ax=ax2, orientation='horizontal',
                       fraction=0.046, pad=0.02, shrink=0.85)
    cb2.set_label('MJy/sr', fontsize=12)

    plt.suptitle(f'{filt} — 2D Polynomial Sky Subtraction (degree={poly_degree})',
                 fontsize=22, y=1.02)
    plt.subplots_adjust(wspace=0.08)
    plt.tight_layout()
    fig.savefig(f'{fig_dir}/{filt}_skysub_comparison.png',
                dpi=dpi, bbox_inches='tight', facecolor='none', transparent=True)
    plt.show()


def plot_background_histogram(data_sub, good_sub, filt, fig_dir='.', dpi=200):
    """
    Histogram of residual pixels after sky subtraction.
    Sanity check: should be centered on ~0 with approximately Gaussian shape.
    """
    from scipy.stats import norm as gaussnorm

    bg_pixels = data_sub[good_sub]
    _, mean, std = sigma_clipped_stats(bg_pixels, sigma=3)

    fig, ax = plt.subplots(figsize=(8, 5), facecolor='none')

    ax.hist(bg_pixels, bins=100, color='steelblue', edgecolor='none',
            alpha=0.8, density=True, label='Residuals')

    x = np.linspace(-4*std, 4*std, 500)
    ax.plot(x, gaussnorm.pdf(x, loc=0, scale=std),
            'r-', lw=2, label=f'Gaussian (σ={std:.3f})')

    ax.axvline(0, color='k', ls='--', lw=1, alpha=0.6)

    ax.text(0.97, 0.95,
            f'mean = {mean:+.4f} MJy/sr\nσ = {std:.3f} MJy/sr',
            transform=ax.transAxes, fontsize=11, va='top', ha='right',
            family='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85))

    ax.set_xlabel('Residual pixel value (MJy/sr)', fontsize=12)
    ax.set_ylabel('Normalized density', fontsize=12)
    ax.set_title(f'{filt} — Residuals after sky subtraction', fontsize=14)
    ax.set_xlim(-4*std, 4*std)
    ax.legend(loc='upper left', fontsize=10)
    plt.tight_layout()
    fig.savefig(f'{fig_dir}/{filt}_background_histogram.png',
                dpi=dpi, bbox_inches='tight', facecolor='none', transparent=True)
    plt.show()

def save_skysub(input_file, data_sub, bg_model, mask, header,
                poly_degree, sigma_upper, grow_npix, segm_file=None,
                output_file=None):
    """
    Save sky-subtracted mosaic with SKYMODEL and SKYMASK extensions.
    Writes processing parameters to the FITS header for reproducibility.
    """
    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = f'{base}_skysub{ext}'

    hdul_out = fits.open(input_file)
    sci_idx = [h.name for h in hdul_out].index('SCI') if 'SCI' in [h.name for h in hdul_out] else 0

    hdul_out[sci_idx].data = data_sub
    hdul_out[sci_idx].header['HISTORY'] = f'Sky subtracted: 2D polynomial degree {poly_degree}'
    hdul_out[sci_idx].header['HISTORY'] = f'Sigma clip = {sigma_upper}, grow = {grow_npix} px'
    hdul_out[sci_idx].header['SKYPOLY'] = (poly_degree, 'Sky polynomial degree')

    if segm_file and os.path.exists(segm_file):
        hdul_out[sci_idx].header['HISTORY'] = f'Segmentation mask: {os.path.basename(segm_file)}'

    bg_hdu = fits.ImageHDU(data=bg_model, header=header, name='SKYMODEL')
    mask_hdu = fits.ImageHDU(data=mask.astype(np.uint8), header=header, name='SKYMASK')
    hdul_out.append(bg_hdu)
    hdul_out.append(mask_hdu)

    hdul_out.writeto(output_file, overwrite=True)
    hdul_out.close()
    print(f'Saved: {output_file}')
    return output_file


def compute_uncertainties(data_sub, good_sub, bkg_cals=None):
    """
    Compute three components of background uncertainty:
    1. Pixel noise (random) — std of background pixels after subtraction
    2. Background level (systematic) — scatter between background exposures
    3. Polynomial bias (systematic) — residual offset from zero

    Parameters
    ----------
    bkg_cals : list of str or None
        Paths to individual background cal files (e.g. Clark GO-3429).
    """
    pixel_noise = np.nanstd(data_sub[good_sub])

    bkg_uncertainty = np.nan
    if bkg_cals is not None and len(bkg_cals) > 1:
        from jwst import datamodels
        bkg_medians = []
        for f in bkg_cals:
            dm = datamodels.open(f)
            _, med, _ = sigma_clipped_stats(dm.data, sigma=3)
            bkg_medians.append(med)
            dm.close()
        bkg_uncertainty = np.std(bkg_medians)

    poly_bias = np.abs(np.nanmedian(data_sub[good_sub]))

    print(f'Pixel noise (random):          {pixel_noise:.3f} MJy/sr')
    if np.isfinite(bkg_uncertainty):
        print(f'Background level (systematic): {bkg_uncertainty:.3f} MJy/sr')
    else:
        print(f'Background level (systematic): N/A')
    print(f'Polynomial bias (systematic):  {poly_bias:.3f} MJy/sr')

    return pixel_noise, bkg_uncertainty, poly_bias

def print_summary(filt, data_sub, good_sub, pixel_noise, bkg_uncertainty, poly_bias,
                  poly_degree, sigma_upper, grow_npix, edge_crop, segm_file, output_file):
    """Print sky subtraction summary."""
    line = '=' * 50
    print(line)
    print(f'  {filt} — 2D Polynomial Sky Subtraction')
    print(line)
    print(f'\n  Background (after subtraction):')
    print(f'    mean = {np.nanmean(data_sub[good_sub]):.6e}')
    print(f'    std  = {np.nanstd(data_sub[good_sub]):.3f} MJy/sr')
    print(f'\n  Uncertainties:')
    print(f'    Pixel noise (random):          {pixel_noise:.3f} MJy/sr')
    if np.isfinite(bkg_uncertainty):
        print(f'    Background level (systematic): {bkg_uncertainty:.3f} MJy/sr')
    else:
        print(f'    Background level (systematic): N/A')
    print(f'    Polynomial bias (systematic):  {poly_bias:.3f} MJy/sr')
    print(f'\n  Settings:')
    print(f'    Polynomial degree:  {poly_degree}')
    print(f'    Sigma clipping:     {sigma_upper}σ')
    print(f'    Mask growth:        {grow_npix} px')
    print(f'    Edge crop:          {edge_crop} px')
    print(f'    Segmentation map:   {os.path.basename(segm_file) if segm_file else "None"}')
    print(f'\n  Output: {os.path.basename(output_file)}')
    print(line)