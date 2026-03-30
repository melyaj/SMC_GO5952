"""
pipeline_utils.py — Shared helpers for JWST/MIRI imaging pipeline
=================================================================
SMC GO-5952 custom data reduction.
Elyajouri Meriem (STScI) - 30/03/2026

Usage:
    sys.path.insert(0, "..")
    from pipeline_utils import *

Functions:
    Data I/O:
        load_mosaic         — Load an i2d FITS mosaic (zeros → NaN)

    Statistics:
        get_stats           — Sigma-clipped median, std, % negative pixels
        stats_label         — Format stats string for plot overlay

    Diagnostics/Figures:
        show_miri           — Quick display of a MIRI image (single panel, no save)
        file_stats          — Print stats for a MIRI image with DQ mask
        plot_mosaic         — Single mosaic with sqrt scaling + colorbar + stats overlay
                              → saves {FILT}_final.png
        plot_comparison     — Side-by-side before/after comparison (2 panels, colorbars, stats)
                              → saves {FILT}_comparison.png
        plot_source_catalog — Mosaic (Greys_r) with red circles on detected sources
                              → saves {FILT}_source_catalog.png

    Pipeline steps:
        run_stage1          — Run calwebb_detector1 + fix_rateints
        run_stage2          — Run calwebb_image2
        fix_rateints_to_rate — Fix rate/rateints averaging bug (Gordon)
        flag_lyot           — Flag Lyot coronagraph artifact in DQ array
        column_clean        — Gordon's cal_column_clean algorithm
        subtract_background — Subtract a master background from cal files
        build_master_background — Median-stack Clark GO-3429 background

    WCS:
        shift_wcs                  — Apply rigid shift to cal file WCS
        analyze_tweakreg_shifts    — Extract and sigma-clip tweakreg shifts per tile
        apply_tile_shifts          — Apply per-tile WCS shifts to all cal files

    Mosaicking:
        run_stage3          — Build association and run calwebb_image3

    Summary:
        print_pipeline_summary — Print pipeline configuration and results

References:
    - Column cleaning: K. Gordon (miri_clean.py)
    - Background field: C. Clark (GO-3429) for F2100W only
"""

import os
import glob
import copy
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import logging

from astropy.io import fits
from astropy.stats import sigma_clipped_stats, sigma_clip
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
    """Load an i2d FITS mosaic, replacing zeros with NaN."""
    with fits.open(fpath) as hdu:
        d = hdu['SCI'].data.astype(float)
        d[d == 0] = np.nan
    return d


# ═══════════════════════════════════════════════════════════
# STATISTICS
# ═══════════════════════════════════════════════════════════

def get_stats(data):
    """Compute sigma-clipped median, std, and % negative pixels."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _, med, std = sigma_clipped_stats(data, sigma=3)
    neg = 100 * np.nansum(data < 0) / np.sum(np.isfinite(data))
    return med, std, neg


def stats_label(med, std, neg):
    """Format stats into a string for plot overlay."""
    return f"median = {med:.3f} MJy/sr\n$\\sigma$ = {std:.3f} MJy/sr\nneg = {neg:.1f}%"


# ═══════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════

def show_miri(data, title='', vmin=None, vmax=None, cmap='afmhot', ax=None):
    """Quick display of a MIRI image."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    if vmin is None or vmax is None:
        norm = simple_norm(data, 'sqrt', percent=99)
    else:
        norm = simple_norm(data, 'sqrt', vmin=vmin, vmax=vmax)
    ax.imshow(data, cmap=cmap, origin='lower', norm=norm)
    ax.set_title(title)
    return ax


def file_stats(data, dq=None, label=''):
    """Print stats for a MIRI image."""
    d = data.copy().astype(float)
    if dq is not None:
        bad = (dq & dqflags.pixel['DO_NOT_USE']) > 0
        d[bad] = np.nan
    d[d == 0] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _, med, std = sigma_clipped_stats(d, sigma=3)
    valid = np.sum(np.isfinite(d))
    total = d.size
    print(f"  {label}: median={med:.3f}  σ={std:.3f}  valid={valid}/{total} ({100*valid/total:.1f}%)")
    return med, std


def plot_mosaic(data, title, filename, fig_dir=".", figsize=(12, 10),
               cmap="afmhot", dpi=200, stats_txt=None):
    """Plot a single mosaic with sqrt scaling, colorbar, and optional stats."""
    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor='white')
    norm = simple_norm(data, 'sqrt', percent=99)
    im = ax.imshow(data, cmap=cmap, origin='lower', norm=norm)
    ax.set_title(title, fontsize=18, pad=10)
    ax.axis('off')

    if stats_txt:
        ax.text(0.03, 0.97, stats_txt, transform=ax.transAxes, fontsize=13,
                va='top', color='white', family='monospace',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.75))

    cb = plt.colorbar(im, ax=ax, orientation='horizontal',
                      fraction=0.046, pad=0.02, shrink=0.85)
    cb.set_label('Surface brightness [MJy/sr]', fontsize=12)
    plt.tight_layout()
    fig.savefig(f"{fig_dir}/{filename}", dpi=dpi, bbox_inches='tight', facecolor='none', transparent=True)
    plt.show()


def plot_comparison(data1, data2, title1, title2, filename, fig_dir=".",
                    figsize=(22, 10), cmap="afmhot", dpi=200,
                    stats1=None, stats2=None):
    """Side-by-side comparison of two mosaics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, facecolor='none')

    for ax, data, title, stats_txt in [(ax1, data1, title1, stats1),
                                        (ax2, data2, title2, stats2)]:
        norm = simple_norm(data, 'sqrt', percent=99)
        im = ax.imshow(data, cmap=cmap, origin='lower', norm=norm)
        ax.set_title(title, fontsize=18, pad=10)
        ax.axis('off')
        if stats_txt:
            ax.text(0.03, 0.97, stats_txt, transform=ax.transAxes, fontsize=13,
                    va='top', color='white', family='monospace',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.75))
        cb = plt.colorbar(im, ax=ax, orientation='horizontal',
                          fraction=0.046, pad=0.02, shrink=0.85)
        cb.set_label('MJy/sr', fontsize=12)

    plt.tight_layout()
    fig.savefig(f"{fig_dir}/{filename}", dpi=dpi, bbox_inches='tight', facecolor='none', transparent=True)
    plt.show()


def plot_source_catalog(stage3_dir, mosaic_data, filt, fig_dir='.', dpi=200):
    """Overlay source catalog on mosaic."""
    from astropy.io import ascii

    cat_file = sorted(glob.glob(f'{stage3_dir}/*_cat.ecsv'))
    if not cat_file:
        print('No catalog found')
        return

    cat = ascii.read(cat_file[0])
    print(f'Catalog: {len(cat)} sources')

    fig, ax = plt.subplots(figsize=(14, 12), facecolor='white')
    show_miri(mosaic_data, cmap='Greys_r', ax=ax)

    flux_col = 'aper_total_flux' if 'aper_total_flux' in cat.colnames else 'flux'
    if flux_col in cat.colnames:
        bright = cat[cat[flux_col] > np.nanpercentile(cat[flux_col], 50)]
    else:
        bright = cat

    ax.scatter(bright['xcentroid'], bright['ycentroid'],
               s=30, facecolors='none', edgecolors='red', linewidths=1, alpha=0.8)
    ax.set_title(f'{filt} — Source catalog ({len(bright)} sources)',
                 fontsize=18)
    ax.axis('off')
    plt.tight_layout()
    fig.savefig(f'{fig_dir}/{filt}_source_catalog.png',
                dpi=dpi, bbox_inches='tight', facecolor='none', transparent=True)
    plt.show()


# ═══════════════════════════════════════════════════════════
# PIPELINE STEPS
# ═══════════════════════════════════════════════════════════

def run_stage1(uncal_files, stage1_dir, ipc_skip=True, jump_threshold=5.0,
               tile1_base='jw05952003001'):
    """
    Run calwebb_detector1 + fix_rateints on all uncal files.

    Returns list of fixed_rate files.
    """
    existing_rate = sorted(glob.glob(f'{stage1_dir}/*_rate.fits'))
    if len(existing_rate) == len(uncal_files):
        print(f'Stage 1 already done: {len(existing_rate)} rate files -- skipping')
    else:
        print(f'Running Stage 1 on {len(uncal_files)} uncal files...')
        for i, f in enumerate(uncal_files):
            tile = 'T1' if tile1_base in f else 'T2'
            print(f'  [{i+1}/{len(uncal_files)}] {tile} -- {os.path.basename(f)}')
            t0 = time.time()
            det1_dict = {
                'ipc': {'skip': ipc_skip},
                'jump': {'rejection_threshold': jump_threshold, 'maximum_cores': 'half'},
                'ramp_fit': {'maximum_cores': 'half'},
            }
            calwebb_detector1.Detector1Pipeline.call(
                f, steps=det1_dict, output_dir=stage1_dir, save_results=True)
            print(f'    done in {(time.time()-t0)/60:.1f} min')

    # fix_rateints
    rate_files = sorted(glob.glob(f'{stage1_dir}/*_rate.fits'))
    old_fixed = glob.glob(f'{stage1_dir}/*_fixed_rate.fits') + \
                glob.glob(f'{stage1_dir}/*_fixed_rateints.fits')
    for f in old_fixed:
        os.remove(f)
    print(f'\nRunning fix_rateints on {len(rate_files)} files...')
    for f in rate_files:
        out = fix_rateints_to_rate(f)
        print(f'  {os.path.basename(f)} -> {os.path.basename(out)}')

    fixed_rate_files = sorted(glob.glob(f'{stage1_dir}/*_fixed_rate.fits'))
    print(f'\n{len(fixed_rate_files)} fixed_rate files ready')
    return fixed_rate_files


def run_stage2(input_files, stage2_dir):
    """Run calwebb_image2 on all rate files. Resample skipped."""
    old_cal = glob.glob(f'{stage2_dir}/*_cal.fits') + \
              glob.glob(f'{stage2_dir}/*_i2d.fits')
    for f in old_cal:
        os.remove(f)

    print(f'Running Stage 2 on {len(input_files)} files...')
    for i, f in enumerate(input_files):
        print(f'  [{i+1}/{len(input_files)}] {os.path.basename(f)}')
        calwebb_image2.Image2Pipeline.call(
            f, steps={'resample': {'skip': True}},
            output_dir=stage2_dir, save_results=True)

    cal_files = sorted(glob.glob(f'{stage2_dir}/*_cal.fits'))
    print(f'{len(cal_files)} cal files ready')
    return cal_files


def fix_rateints_to_rate(mfile):
    """
    Fix the rate/rateints averaging bug (Gordon).
    Re-averages the rateints file with proper NaN handling.
    """
    rifile = mfile.replace("rate", "rateints")
    rdata = datamodels.open(mfile)
    ridata = datamodels.open(rifile)

    ridata.data[ridata.data == 0.0] = np.nan
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", message="Mean of empty slice")
        rdata.data = np.nanmean(ridata.data, axis=0)

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
    """Flag the Lyot coronagraph artifact region in the DQ array."""
    dm = datamodels.open(cal_file)
    dm.dq[lyot_row:, :lyot_col] |= (dqflags.pixel['NON_SCIENCE'] |
                                      dqflags.pixel['DO_NOT_USE'])
    dm.save(cal_file)
    dm.close()


def column_clean(cal_file, sigma=20, exclude_above=250.0):
    """
    Gordon's cal_column_clean algorithm.
    Removes column-correlated noise (pulldown/pullup).
    Must be applied BEFORE background subtraction.
    """
    dm = datamodels.open(cal_file)
    rimage = copy.deepcopy(dm.data)
    kernel = Gaussian1DKernel(stddev=sigma)

    bdata = (dm.dq & dqflags.pixel['DO_NOT_USE']) > 0
    rimage[bdata] = np.nan
    rimage[rimage == 0.0] = np.nan
    rimage[rimage > exclude_above] = np.nan

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='All-NaN slice encountered')
        colmeds = np.nanmedian(rimage, axis=0)

    colmeds_smooth = convolve(colmeds - np.nanmedian(colmeds), kernel)
    colmeds_sub = colmeds - colmeds_smooth

    colimage = np.tile(colmeds_sub, (dm.data.shape[0], 1))
    colimage[bdata] = np.nan
    colimage -= np.nanmedian(colimage)
    colimage[bdata] = 0.0
    colimage = np.nan_to_num(colimage, nan=0.0)

    dm.data -= colimage
    dm.save(cal_file)
    dm.close()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _, _, correction_std = sigma_clipped_stats(colimage[colimage != 0], sigma=3)

    return correction_std


def subtract_background(cal_file, master_bkg, output_dir, suffix="_bkgsub"):
    """Subtract a master background frame from a cal file."""
    dm = datamodels.open(cal_file)
    dm.data -= master_bkg
    outfile = os.path.join(
        output_dir,
        os.path.basename(cal_file).replace('_cal.fits', f'{suffix}_cal.fits')
    )
    dm.save(outfile)
    dm.close()
    return outfile


def build_master_background(bkg_dir, pattern="jw03429*/jw*_cal.fits"):
    """Median-stack Clark's GO-3429 background observations."""
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


# ═══════════════════════════════════════════════════════════
# WCS
# ═══════════════════════════════════════════════════════════

def shift_wcs(mfile, shifts, out_suffix):
    """Apply a rigid RA/Dec shift to the WCS of a cal file."""
    dm = datamodels.open(mfile)
    wcs_c = JWSTWCSCorrector(wcs=dm.meta.wcs, wcsinfo=dm.meta.wcsinfo.instance)
    wcs_c.set_correction(matrix=[[1, 0], [0, 1]], shift=shifts, ref_tpwcs=wcs_c)
    dm.meta.wcs = wcs_c.wcs
    outfile = mfile.replace("_cal.fits", f"{out_suffix}_cal.fits")
    dm.save(outfile)
    dm.close()
    return outfile


def analyze_tweakreg_shifts(stage3_dir, n_images_per_tile=4):
    """
    Extract and sigma-clip tweakreg shifts per tile.
    Returns (tile1_shifts, tile2_shifts) as [V2, V3] in arcsec.
    """
    RAD2ARCSEC = 3600.0 * np.rad2deg(1.0)

    twfiles = np.sort(glob.glob(f'{stage3_dir}/*_tweakreg.fits'))
    print(f'Found {len(twfiles)} tweakreg files\n')

    shifts = np.zeros((2, len(twfiles)))
    for k, cfile in enumerate(twfiles):
        aligned_model = datamodels.open(cfile)
        try:
            cshift = RAD2ARCSEC * aligned_model.meta.wcs.forward_transform['tp_affine'].translation.value
            shifts[:, k] = cshift
            print(f'  {os.path.basename(cfile)}: {cshift}')
        except Exception:
            print(f'  {os.path.basename(cfile)}: [0.0, 0.0] (reference)')
        aligned_model.close()

    tile_shifts = []
    for k in range(2):
        k1 = k * n_images_per_tile
        k2 = k1 + n_images_per_tile
        avex = float(np.mean(sigma_clip(shifts[0, k1:k2])))
        avey = float(np.mean(sigma_clip(shifts[1, k1:k2])))
        tile_shifts.append([avex, avey])
        print(f'Tile {k+1}: V2 = {avex:+.4f}",  V3 = {avey:+.4f}"')

    return tile_shifts[0], tile_shifts[1]


def apply_tile_shifts(cal_files, tile1_shifts, tile2_shifts,
                      tile1_base='jw05952003001'):
    """Apply per-tile WCS shifts to all cal files. Returns list of wcs_cal files."""
    print(f'Applying shifts:')
    print(f'  T1: {tile1_shifts}')
    print(f'  T2: {tile2_shifts}\n')

    stage2_dir = os.path.dirname(cal_files[0])
    for f in glob.glob(f'{stage2_dir}/*_wcs_cal.fits'):
        os.remove(f)

    for cfile in cal_files:
        if tile1_base in cfile:
            s, tile = tile1_shifts, 'T1'
        else:
            s, tile = tile2_shifts, 'T2'
        outfile = shift_wcs(cfile, s, "_wcs")
        print(f'  {tile}: {os.path.basename(outfile)}')

    wcs_cal_files = sorted(glob.glob(f'{stage2_dir}/*_wcs_cal.fits'))
    print(f'\n{len(wcs_cal_files)} WCS-corrected cal files')
    return wcs_cal_files


# ═══════════════════════════════════════════════════════════
# MOSAICKING
# ═══════════════════════════════════════════════════════════

def run_stage3(input_files, name, stage3_dir, tweakreg=False, skymatch=True,
               sky_subtract=False, outlier_det=False, pixel_scale=0.11,
               pixfrac=0.8, kernel="square", extra_steps=None):
    """Run calwebb_image3 (Stage 3) with custom settings."""
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


# ═══════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════

def print_pipeline_summary(target, program, filt, m_ours=None,
                           tile1_shifts=None, tile2_shifts=None,
                           ipc_skip=True, jump_threshold=5.0,
                           lyot_row=700, lyot_col=310,
                           pixel_scale=0.11, pixfrac=1.0, kernel='square',
                           tweakreg=False, skymatch=False,
                           sky_subtract=False, outlier_det=False):
    """Print pipeline configuration and results."""
    line = '=' * 50
    print(line)
    print(f'  {target} ({program}) — {filt}')
    print(line)
    if m_ours:
        print(f'\n  Result:')
        print(f'    median = {m_ours[0]:.3f} MJy/sr')
        print(f'    σ = {m_ours[1]:.3f} MJy/sr')
    if tile1_shifts:
        print(f'\n  WCS shifts:')
        print(f'    Tile 1: {tile1_shifts}')
        print(f'    Tile 2: {tile2_shifts}')
    print(f'\n  Stage 1:')
    print(f'    IPC:              {"SKIP" if ipc_skip else "ON"}')
    print(f'    Jump threshold:   {jump_threshold}σ')
    print(f'\n  Lyot flag: row>{lyot_row}, col<{lyot_col}')
    print(f'\n  Stage 3:')
    print(f'    Pixel scale:      {pixel_scale} arcsec/pix')
    print(f'    Pixfrac:          {pixfrac}')
    print(f'    Kernel:           {kernel}')
    print(f'    Tweakreg:         {"ON" if tweakreg else "OFF"}')
    print(f'    Skymatch:         {"ON" if skymatch else "OFF"}')
    print(f'    Sky subtract:     {"ON" if sky_subtract else "OFF"}')
    print(f'    Outlier det:      {"ON" if outlier_det else "OFF"}')
    print(line)
