"""
pipeline_utils.py — Shared helpers for JWST/MIRI imaging pipeline
=================================================================
SMC GO-5952 custom data reduction.

Usage:
    sys.path.insert(0, "/Users/melyajou/SMC_GO5952/miri")
    from pipeline_utils import *

Functions:
    Data I/O:
        load_mosaic         — Load an i2d FITS mosaic (zeros → NaN)

    Statistics:
        get_stats           — Sigma-clipped median, std, % negative pixels
        stats_label         — Format stats string for plot overlay

    Plotting:
        plot_mosaic         — Single mosaic with sqrt scaling + colorbar
        plot_comparison     — Side-by-side before/after comparison

    Pipeline steps:
        fix_rateints_to_rate — Fix rate/rateints averaging bug (Gordon)
        shift_wcs           — Apply rigid RA/Dec shift to cal file WCS
        flag_lyot           — Flag Lyot coronagraph artifact in DQ array
        column_clean        — Gordon's cal_column_clean algorithm
        subtract_background — Subtract a master background from cal files
        run_stage3          — Build association and run calwebb_image3

References:
    - Column cleaning: K. Gordon (miri_clean.py)
    - Background field: C. Clark (GO-3429)
    - Clark+2021, ApJ, 921, 35
"""

import os
import glob
import copy
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import logging

from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.visualization import simple_norm
from astropy.convolution import Gaussian1DKernel, convolve
from tweakwcs import JWSTWCSCorrector

from jwst import datamodels
from jwst.datamodels import dqflags
from jwst.pipeline import calwebb_detector1, calwebb_image2, calwebb_image3
from jwst.associations import asn_from_list
from jwst.associations.lib.rules_level3_base import DMS_Level3_Base

# Suppress verbose pipeline logging
logging.getLogger('stpipe').setLevel(logging.WARNING)
logging.getLogger('CRDS').setLevel(logging.WARNING)


# ═══════════════════════════════════════════════════════════
# DATA I/O
# ═══════════════════════════════════════════════════════════

def load_mosaic(fpath):
    """
    Load an i2d FITS mosaic, replacing zeros with NaN.

    Parameters
    ----------
    fpath : str
        Path to the i2d FITS file.

    Returns
    -------
    data : 2D ndarray
        The SCI extension data with zeros replaced by NaN.
    """
    with fits.open(fpath) as hdu:
        d = hdu['SCI'].data.astype(float)
        d[d == 0] = np.nan
    return d


# ═══════════════════════════════════════════════════════════
# STATISTICS
# ═══════════════════════════════════════════════════════════

def get_stats(data):
    """
    Compute sigma-clipped statistics on an image.

    Parameters
    ----------
    data : 2D ndarray

    Returns
    -------
    med : float
        Sigma-clipped median.
    std : float
        Sigma-clipped standard deviation.
    neg : float
        Percentage of finite pixels that are negative.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _, med, std = sigma_clipped_stats(data, sigma=3)
    neg = 100 * np.nansum(data < 0) / np.sum(np.isfinite(data))
    return med, std, neg


def stats_label(med, std, neg):
    """Format stats into a string for plot overlay."""
    return f"$\\sigma$ = {std:.3f} MJy/sr\nneg = {neg:.1f}%"


# ═══════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════

def plot_mosaic(data, title, filename, fig_dir=".", figsize=(12, 10),
               cmap="afmhot", dpi=200, stats_txt=None):
    """
    Plot a single mosaic with sqrt scaling, colorbar, and optional stats.

    Parameters
    ----------
    data : 2D ndarray
        Image data to plot.
    title : str
        Plot title.
    filename : str
        Output filename (saved in fig_dir/).
    fig_dir : str
        Directory to save the figure.
    figsize : tuple
        Figure size.
    cmap : str
        Colormap name.
    dpi : int
        Output resolution.
    stats_txt : str or None
        Optional text overlay (e.g., from stats_label()).
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor='white')
    norm = simple_norm(data, 'sqrt', percent=99)
    im = ax.imshow(data, cmap=cmap, origin='lower', norm=norm)
    ax.set_title(title, fontsize=18, fontweight='bold', pad=10)
    ax.axis('off')

    if stats_txt:
        ax.text(0.03, 0.97, stats_txt, transform=ax.transAxes, fontsize=13,
                va='top', color='white', family='monospace',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.75))

    cb = plt.colorbar(im, ax=ax, orientation='horizontal',
                      fraction=0.046, pad=0.02, shrink=0.85)
    cb.set_label('Surface brightness [MJy/sr]', fontsize=12)
    plt.tight_layout()
    fig.savefig(f"{fig_dir}/{filename}", dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.show()


def plot_comparison(data1, data2, title1, title2, filename, fig_dir=".",
                    figsize=(22, 10), cmap="afmhot", dpi=200,
                    stats1=None, stats2=None):
    """
    Side-by-side comparison of two mosaics.

    Parameters
    ----------
    data1, data2 : 2D ndarray
        Left and right images.
    title1, title2 : str
        Titles for each panel.
    filename : str
        Output filename.
    fig_dir : str
        Directory to save the figure.
    stats1, stats2 : str or None
        Optional stats text for each panel.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, facecolor='white')

    for ax, data, title, stats_txt in [(ax1, data1, title1, stats1),
                                        (ax2, data2, title2, stats2)]:
        norm = simple_norm(data, 'sqrt', percent=99)
        im = ax.imshow(data, cmap=cmap, origin='lower', norm=norm)
        ax.set_title(title, fontsize=18, fontweight='bold', pad=10)
        ax.axis('off')
        if stats_txt:
            ax.text(0.03, 0.97, stats_txt, transform=ax.transAxes, fontsize=13,
                    va='top', color='white', family='monospace',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.75))
        cb = plt.colorbar(im, ax=ax, orientation='horizontal',
                          fraction=0.046, pad=0.02, shrink=0.85)
        cb.set_label('MJy/sr', fontsize=12)

    plt.tight_layout()
    fig.savefig(f"{fig_dir}/{filename}", dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.show()


# ═══════════════════════════════════════════════════════════
# PIPELINE STEPS
# ═══════════════════════════════════════════════════════════

def fix_rateints_to_rate(mfile):
    """
    Fix the rate/rateints averaging bug (Gordon).

    The default JWST pipeline averages integrations incorrectly.
    This re-averages the rateints file with proper NaN handling
    to produce a corrected rate file.

    Parameters
    ----------
    mfile : str
        Path to the *_rate.fits file.

    Returns
    -------
    nfile_r : str
        Path to the corrected *_fixed_rate.fits file.
    """
    rifile = mfile.replace("rate", "rateints")
    rdata = datamodels.open(mfile)
    ridata = datamodels.open(rifile)

    # Replace zeros with NaN before averaging
    ridata.data[ridata.data == 0.0] = np.nan
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", message="Mean of empty slice")
        rdata.data = np.nanmean(ridata.data, axis=0)

    # Flag NaN pixels in the DQ array
    ndata = np.isnan(rdata.data)
    rdata.data[ndata] = 0.0
    rdata.dq[ndata] = 3

    ridata.data[np.isnan(ridata.data)] = 0.0

    nfile_ri = mfile.replace("rate.fits", "fixed_rateints.fits")
    ridata.save(nfile_ri)
    nfile_r = mfile.replace("rate.fits", "fixed_rate.fits")
    rdata.save(nfile_r)
    return nfile_r


def flag_lyot(cal_file, lyot_row=700, lyot_col=310):
    """
    Flag the Lyot coronagraph artifact region in the DQ array.

    The MIRI imager has a Lyot coronagraph that leaves an artifact
    in the upper-left corner of the detector. This flags those pixels
    as DO_NOT_USE so they are excluded from mosaicking.

    Parameters
    ----------
    cal_file : str
        Path to a *_cal.fits file. Modified in-place.
    lyot_row : int
        Row above which the artifact appears.
    lyot_col : int
        Column to the left of which the artifact appears.
    """
    dm = datamodels.open(cal_file)
    dm.dq[lyot_row:, :lyot_col] |= (dqflags.pixel['NON_SCIENCE'] |
                                      dqflags.pixel['DO_NOT_USE'])
    dm.save(cal_file)
    dm.close()


def column_clean(cal_file, sigma=20, exclude_above=250.0):
    """
    Gordon's cal_column_clean algorithm.

    Removes column-correlated noise (pulldown/pullup) by:
    1. Computing the median of each column (excluding bad/bright pixels)
    2. Smoothing that 1D profile with a Gaussian kernel
    3. Subtracting the smoothed profile → isolates high-frequency column noise
    4. Tiling back to 2D and subtracting from the image

    IMPORTANT: Must be applied to cal files BEFORE background subtraction,
    because the column medians need the full background level to work correctly.

    Parameters
    ----------
    cal_file : str
        Path to a *_cal.fits file. Modified in-place.
    sigma : float
        Gaussian σ (in pixels) for smoothing the column median profile.
    exclude_above : float
        Exclude pixels above this value (MJy/sr) from column medians.
        Filter-dependent — tune based on background level.

    Returns
    -------
    correction_std : float
        Standard deviation of the correction image (diagnostic).
    """
    dm = datamodels.open(cal_file)
    rimage = copy.deepcopy(dm.data)
    kernel = Gaussian1DKernel(stddev=sigma)

    # Mask bad pixels (DQ) + zeros + bright sources
    bdata = (dm.dq & dqflags.pixel['DO_NOT_USE']) > 0
    rimage[bdata] = np.nan
    rimage[rimage == 0.0] = np.nan
    rimage[rimage > exclude_above] = np.nan

    # Column medians (ignoring masked pixels)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='All-NaN slice encountered')
        colmeds = np.nanmedian(rimage, axis=0)

    # Smooth the column medians with a Gaussian kernel.
    # Subtracting the smoothed version isolates the high-frequency
    # column-to-column noise we want to remove.
    colmeds_smooth = convolve(colmeds - np.nanmedian(colmeds), kernel)
    colmeds_sub = colmeds - colmeds_smooth

    # Build 2D correction image
    colimage = np.tile(colmeds_sub, (dm.data.shape[0], 1))
    colimage[bdata] = np.nan
    colimage -= np.nanmedian(colimage)
    colimage[bdata] = 0.0
    colimage = np.nan_to_num(colimage, nan=0.0)

    # Apply correction
    dm.data -= colimage
    dm.save(cal_file)
    dm.close()

    # Diagnostic: std of the correction
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _, _, correction_std = sigma_clipped_stats(colimage[colimage != 0], sigma=3)

    return correction_std


def subtract_background(cal_file, master_bkg, output_dir, suffix="_bkgsub"):
    """
    Subtract a master background frame from a cal file.

    Parameters
    ----------
    cal_file : str
        Path to a *_cal.fits file.
    master_bkg : 2D ndarray
        Background frame (e.g., median of Clark GO-3429 cals).
    output_dir : str
        Directory for the output file.
    suffix : str
        Suffix inserted before _cal.fits in the output filename.

    Returns
    -------
    outfile : str
        Path to the background-subtracted output file.
    """
    dm = datamodels.open(cal_file)
    dm.data -= master_bkg
    outfile = os.path.join(
        output_dir,
        os.path.basename(cal_file).replace('_cal.fits', f'{suffix}_cal.fits')
    )
    dm.save(outfile)
    dm.close()
    return outfile


def shift_wcs(mfile, shifts, out_suffix):
    """
    Apply a rigid RA/Dec shift to the WCS of a cal file.

    Shifts are measured from F560W alignment to Gaia.
    Uses tweakwcs to update the WCS without modifying pixel data.

    Parameters
    ----------
    mfile : str
        Path to the input *_cal.fits file.
    shifts : list of float
        [delta_RA, delta_Dec] in arcsec.
    out_suffix : str
        Suffix appended before _cal.fits in the output filename.

    Returns
    -------
    outfile : str
        Path to the WCS-shifted output file.
    """
    dm = datamodels.open(mfile)
    wcs_c = JWSTWCSCorrector(wcs=dm.meta.wcs, wcsinfo=dm.meta.wcsinfo.instance)
    wcs_c.set_correction(matrix=[[1, 0], [0, 1]], shift=shifts, ref_tpwcs=wcs_c)
    dm.meta.wcs = wcs_c.wcs
    outfile = mfile.replace("_cal.fits", f"{out_suffix}_cal.fits")
    dm.save(outfile)
    dm.close()
    return outfile


def run_stage3(input_files, name, stage3_dir, tweakreg=False, skymatch=True,
               sky_subtract=False, outlier_det=False, pixel_scale=0.11,
               pixfrac=0.8, kernel="square", extra_steps=None):
    """
    Run calwebb_image3 (Stage 3) with custom settings.

    Builds a Level 3 association from the input files and runs
    the mosaic pipeline.

    Parameters
    ----------
    input_files : list of str
        Paths to the cal files to mosaic.
    name : str
        Product name for the association and output files.
    stage3_dir : str
        Output directory.
    tweakreg : bool
        Align dithers (skip if WCS shifts already applied).
    skymatch : bool
        Match sky levels between exposures.
        NOTE: unreliable if overlap regions contain bright extended emission.
    sky_subtract : bool
        Subtract matched sky (False = offset only).
    outlier_det : bool
        Enable outlier pixel rejection.
    pixel_scale : float
        Output pixel scale in arcsec/pix.
    pixfrac : float
        Drizzle pixel fraction.
    kernel : str
        Drizzle kernel ("square" or "gaussian").
    extra_steps : dict or None
        Additional step overrides.

    Returns
    -------
    i2d_file : str
        Path to the output *_i2d.fits mosaic.
    """
    # Clean previous outputs
    for f in glob.glob(f"{stage3_dir}/{name}*"):
        os.remove(f)

    # Build association
    asn = asn_from_list.asn_from_list(
        input_files, rule=DMS_Level3_Base, product_name=name
    )
    asn_file = f"{stage3_dir}/{name}.json"
    with open(asn_file, 'w') as fp:
        _, s = asn.dump(format='json')
        fp.write(s)

    # Pipeline configuration
    steps = {
        "tweakreg":          {"skip": not tweakreg},
        "skymatch":          {"skymethod": "match", "subtract": sky_subtract, "skip": not skymatch},
        "outlier_detection": {"skip": not outlier_det},
        "resample":          {"kernel": kernel, "pixel_scale": pixel_scale, "pixfrac": pixfrac},
    }
    if extra_steps:
        steps.update(extra_steps)

    calwebb_image3.Image3Pipeline.call(
        asn_file, steps=steps, output_dir=stage3_dir, save_results=True
    )
    return sorted(glob.glob(f"{stage3_dir}/{name}*_i2d.fits"))[0]


# ─── Directory setup helper ──────────────────────────────

def build_master_background(bkg_dir, pattern="jw03429*/jw*_cal.fits"):
    """
    Build a master background from Clark's GO-3429 observations.

    Median-stacks all background cal files into a single 2D frame.

    Parameters
    ----------
    bkg_dir : str
        Base directory containing background cal files.
    pattern : str
        Glob pattern to find the cal files.

    Returns
    -------
    master_bkg : 2D ndarray
        Median-stacked background frame.
    n_files : int
        Number of files used.
    """
    bkg_cals = sorted(glob.glob(f"{bkg_dir}/{pattern}"))
    if not bkg_cals:
        raise FileNotFoundError(f"No background files found: {bkg_dir}/{pattern}")

    bkg_stack = []
    for f in bkg_cals:
        dm = datamodels.open(f)
        bkg_stack.append(dm.data.astype(float))
        dm.close()

    master_bkg = np.nanmedian(bkg_stack, axis=0)
    return master_bkg, len(bkg_cals)
