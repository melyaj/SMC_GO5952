"""
nircam_pipeline_utils.py — SMC GO-5952 NIRCam reduction utilities
=================================================================
Companion to NRC_pipeline.ipynb.

8 filters:
  LW broadband : F300M, F335M, F360M, F444W   (nrcalong/nrcblong, 0.042"/px)
  SW broadband : F150W, F200W                  (8 SW detectors,    0.021"/px)
  SW narrow    : F187N, F212N                  (8 SW detectors,    0.021"/px)
"""

import os, glob, time, shutil, warnings, logging
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.visualization import simple_norm
from jwst import datamodels
from jwst.datamodels import dqflags
from jwst.pipeline import calwebb_detector1, calwebb_image2, calwebb_image3
from jwst.associations import asn_from_list
from jwst.associations.lib.rules_level3_base import DMS_Level3_Base


# ═══════════════════════════════════════════════════════════════════════
#  Filter config
# ═══════════════════════════════════════════════════════════════════════

SHORT_FILTS  = {'F150W', 'F187N', 'F200W', 'F212N'}
LONG_FILTS   = {'F300M', 'F335M', 'F360M', 'F444W'}
NARROW_FILTS = {'F187N', 'F212N'}
ALL_FILTS    = SHORT_FILTS | LONG_FILTS

def filter_config(filt):
    """Return dict: is_short, is_narrow, pixel_scale, det_list, det_a, det_b"""
    if filt not in ALL_FILTS:
        raise ValueError(f"Unknown filter {filt}. Choose from {sorted(ALL_FILTS)}")
    is_short  = filt in SHORT_FILTS
    is_narrow = filt in NARROW_FILTS
    pixel_scale = 0.021 if is_short else 0.042
    det_list = (['nrca1','nrca2','nrca3','nrca4',
                 'nrcb1','nrcb2','nrcb3','nrcb4']
                if is_short else ['nrcalong','nrcblong'])
    det_a = 'nrca1' if is_short else 'nrcalong'
    det_b = 'nrcb1' if is_short else 'nrcblong'
    return dict(is_short=is_short, is_narrow=is_narrow,
                pixel_scale=pixel_scale, det_list=det_list,
                det_a=det_a, det_b=det_b)


def setup_dirs(base_dir, filt, is_narrow=False):
    """Create and return directory tree."""
    workdir = os.path.join(base_dir, filt)
    dirs = dict(
        workdir  = workdir,
        stage0   = os.path.join(workdir, 'stage0'),
        stage1   = os.path.join(workdir, 'stage1'),
        stage2   = os.path.join(workdir, 'stage2'),
        stage3   = os.path.join(workdir, 'stage3'),
        diag     = os.path.join(workdir, 'diagnostics'),
        mast_ref = os.path.join(workdir, 'mast_reference'),
    )
    if is_narrow:
        dirs['stage2_eq'] = os.path.join(workdir, 'stage2_equalized')
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


# ═══════════════════════════════════════════════════════════════════════
#  Download helpers
# ═══════════════════════════════════════════════════════════════════════

def download_uncals(filt, stage0, base_dir, is_narrow=False):
    """Download Level-1 uncal files from MAST if not present."""
    existing = sorted(glob.glob(os.path.join(stage0, '*_uncal.fits')))
    if existing:
        print(f'Already have {len(existing)} uncal files — skipping')
        return existing
    from astroquery.mast import Observations
    token = os.environ.get('MAST_TOKEN') or input('MAST token: ')
    Observations.login(token=token)
    obs = Observations.query_criteria(obs_collection='JWST', proposal_id='5952',
                                      instrument_name='NIRCAM/IMAGE')
    obs_filt = (obs[[filt in str(f) for f in obs['filters']]] if is_narrow
                else obs[obs['filters'] == filt])
    products = Observations.get_product_list(obs_filt)
    filtered = Observations.filter_products(products, calib_level=[1],
                                            productType='SCIENCE', extension='fits')
    dl_dir = os.path.join(base_dir, f'dl_tmp_{filt}')
    shutil.rmtree(dl_dir, ignore_errors=True)
    for attempt in range(3):
        try:
            Observations.download_products(filtered, download_dir=dl_dir); break
        except Exception as e:
            print(f'  Attempt {attempt+1} failed: {e}'); time.sleep(30)
    for f in glob.glob(os.path.join(dl_dir, 'mastDownload/JWST/*/*uncal.fits')):
        shutil.copy(f, os.path.join(stage0, os.path.basename(f)))
    shutil.rmtree(dl_dir, ignore_errors=True)
    return sorted(glob.glob(os.path.join(stage0, '*_uncal.fits')))


def download_mast_i2d(filt, mast_ref, base_dir, is_narrow=False):
    """Download MAST Level-3 i2d if not present."""
    existing = sorted(glob.glob(os.path.join(mast_ref, '*_i2d.fits')))
    if existing:
        print(f'Already have {len(existing)} MAST i2d — skipping')
        return existing
    from astroquery.mast import Observations
    obs = Observations.query_criteria(obs_collection='JWST', proposal_id='5952',
                                      instrument_name='NIRCAM/IMAGE')
    obs_filt = (obs[[filt in str(f) for f in obs['filters']]] if is_narrow
                else obs[obs['filters'] == filt])
    products = Observations.get_product_list(obs_filt)
    filtered = Observations.filter_products(products, calib_level=[3],
                                            productSubGroupDescription='I2D', extension='fits')
    if len(filtered) == 0:
        print('No MAST i2d found'); return []
    dl_dir = os.path.join(base_dir, f'dl_tmp_{filt}_i2d')
    shutil.rmtree(dl_dir, ignore_errors=True)
    Observations.download_products(filtered, download_dir=dl_dir)
    for f in glob.glob(os.path.join(dl_dir, 'mastDownload/JWST/*/*_i2d.fits')):
        shutil.copy(f, os.path.join(mast_ref, os.path.basename(f)))
    shutil.rmtree(dl_dir, ignore_errors=True)
    return sorted(glob.glob(os.path.join(mast_ref, '*_i2d.fits')))


# ═══════════════════════════════════════════════════════════════════════
#  Pipeline stages
# ═══════════════════════════════════════════════════════════════════════

def run_stage1(uncal_files, stage1, clean_flicker_noise=True):
    """calwebb_detector1.  Returns list of rate paths."""
    steps = {
        'ramp_fit':            {'suppress_one_group': False, 'maximum_cores': 'half'},
        'jump':                {'expand_large_events': True,  'maximum_cores': 'half'},
        'clean_flicker_noise': {'skip': not clean_flicker_noise},
    }
    for i, f in enumerate(uncal_files):
        det = fits.getheader(f, 0).get('DETECTOR', '?').lower()
        tag = ' ← science' if 'nrcb' in det else ''
        print(f'[{i+1}/{len(uncal_files)}] {det}{tag}')
        t0 = time.time()
        calwebb_detector1.Detector1Pipeline.call(
            f, steps=steps, output_dir=stage1, save_results=True)
        print(f'  {(time.time()-t0)/60:.1f} min')
    out = sorted(glob.glob(os.path.join(stage1, '*_rate.fits')))
    print(f'\n{len(out)} rate files')
    return out


def run_stage2(rate_files, stage2):
    """calwebb_image2 (resample skipped).  Returns list of cal paths."""
    for i, f in enumerate(rate_files):
        det = fits.getheader(f, 1).get('DETECTOR', '?').lower()
        print(f'[{i+1}/{len(rate_files)}] {det}')
        calwebb_image2.Image2Pipeline.call(
            f, steps={'resample': {'skip': True}},
            output_dir=stage2, save_results=True)
    out = sorted(glob.glob(os.path.join(stage2, '*_cal.fits')))
    print(f'\n{len(out)} cal files')
    return out


def equalize_detectors(cal_files, stage2_eq):
    """Per-detector scalar equalization (narrow-band only).
    Subtracts (median - global_min) per file.  Returns corrected paths."""
    print(f'Equalizing {len(cal_files)} files ...')
    meds = {}
    for f in cal_files:
        dm = datamodels.open(f)
        bad = (dm.dq & dqflags.pixel['DO_NOT_USE']) > 0
        d = dm.data.astype(float); d[bad | (d == 0)] = np.nan
        _, med, _ = sigma_clipped_stats(d[np.isfinite(d)], sigma=3)
        meds[f] = med
        det = dm.meta.instrument.detector.lower()
        print(f'  {det}  med={med:.4f}'); dm.close()

    ref = min(meds.values())
    print(f'Reference: {ref:.4f} MJy/sr')

    # Clean old equalized files to avoid mixing runs
    os.makedirs(stage2_eq, exist_ok=True)
    for old in glob.glob(os.path.join(stage2_eq, '*.fits')):
        os.remove(old)

    out = []
    for f, med in meds.items():
        offset = med - ref
        outname = os.path.basename(f)              # same name, different dir
        outfile = os.path.join(stage2_eq, outname)
        dm = datamodels.open(f)
        dm.data = (dm.data.astype(float) - offset).astype(np.float32)
        dm.save(outfile); dm.close()
        print(f'  offset={offset:+.4f}  → {outname}')
        out.append(outfile)
    out = sorted(out)
    print(f'{len(out)} equalized files')
    return out

def equalize_detectors_masked(cal_files, stage2_eq, nsigma=2.5):
    """Equalize with emission masking — safer for narrow-band."""
    print(f'Equalizing {len(cal_files)} files (masked, {nsigma}σ) ...')
    os.makedirs(stage2_eq, exist_ok=True)
    for old in glob.glob(os.path.join(stage2_eq, '*.fits')):
        os.remove(old)

    meds = {}
    for f in cal_files:
        dm = datamodels.open(f)
        d = dm.data.astype(float)
        bad = (dm.dq & dqflags.pixel['DO_NOT_USE']) > 0
        d[bad | (d == 0)] = np.nan
        _, med_rough, std_rough = sigma_clipped_stats(d[np.isfinite(d)], sigma=3)
        bg_only = np.isfinite(d) & (d < med_rough + nsigma * std_rough)
        _, med_clean, _ = sigma_clipped_stats(d[bg_only], sigma=3)
        meds[f] = med_clean
        det = dm.meta.instrument.detector.lower()
        print(f'  {det}  med={med_clean:.4f}'); dm.close()

    ref = min(meds.values())
    print(f'Reference: {ref:.4f} MJy/sr')
    out = []
    for f, med in meds.items():
        offset = med - ref
        outfile = os.path.join(stage2_eq, os.path.basename(f))
        dm = datamodels.open(f)
        dm.data = (dm.data.astype(float) - offset).astype(np.float32)
        dm.save(outfile); dm.close()
        print(f'  offset={offset:+.4f}'); out.append(outfile)
    print(f'{len(out)} equalized files')
    return sorted(out)

def equalize_detectors_refA(cal_files, stage2_eq):
    """Equalize nrcb using nrca background (no emission bias)."""
    print(f'Equalizing {len(cal_files)} files (module A reference) ...')
    os.makedirs(stage2_eq, exist_ok=True)
    for old in glob.glob(os.path.join(stage2_eq, '*.fits')):
        os.remove(old)

    # Measure all medians + group by dither
    info = {}
    for f in cal_files:
        dm = datamodels.open(f)
        bad = (dm.dq & dqflags.pixel['DO_NOT_USE']) > 0
        d = dm.data.astype(float); d[bad | (d == 0)] = np.nan
        _, med, _ = sigma_clipped_stats(d[np.isfinite(d)], sigma=3)
        det = dm.meta.instrument.detector.lower()
        dither = dm.meta.observation.exposure_number
        info[f] = {'med': med, 'det': det, 'dither': dither}
        print(f'  {det}  dither={dither}  med={med:.4f}')
        dm.close()

    # For each dither: compute nrca average level
    dithers = set(v['dither'] for v in info.values())
    nrca_level = {}
    for dith in dithers:
        nrca_meds = [v['med'] for v in info.values()
                     if v['dither'] == dith and 'nrca' in v['det']]
        nrca_level[dith] = np.median(nrca_meds)
        print(f'  Dither {dith}: nrca level = {nrca_level[dith]:.4f}')

    # Reference = global minimum of nrca levels
    ref = min(nrca_level.values())
    print(f'Reference: {ref:.4f} MJy/sr')

    # Apply offsets
    out = []
    for f, v in info.items():
        if 'nrca' in v['det']:
            offset = v['med'] - ref           # nrca: use own median
        else:
            offset = nrca_level[v['dither']] - ref  # nrcb: use nrca level
        outfile = os.path.join(stage2_eq, os.path.basename(f))
        dm = datamodels.open(f)
        dm.data = (dm.data.astype(float) - offset).astype(np.float32)
        dm.save(outfile); dm.close()
        print(f'  {v["det"]}  d={v["dither"]}  offset={offset:+.4f}')
        out.append(outfile)

    print(f'{len(out)} equalized files')
    return sorted(out)
    
def run_stage3(cal_files, stage3, pixel_scale, filt,
               tweakreg=True, skymatch=False, sky_subtract=False,
               outlier_detection=True):
    """calwebb_image3.  Returns list of i2d paths."""
    for f in glob.glob(os.path.join(stage3, '*')):
        os.remove(f)

    asn_name = f'nircam_{filt}_final'
    asn = asn_from_list.asn_from_list(
        cal_files, rule=DMS_Level3_Base, product_name=asn_name)
    asn_file = os.path.join(stage3, f'{asn_name}.json')
    with open(asn_file, 'w') as fp:
        _, s = asn.dump(format='json'); fp.write(s)

    if skymatch:
        sky_step = {'skip': False, 'skymethod': 'match',
                    'subtract': sky_subtract}
    else:
        sky_step = {'skip': True}

    steps = {
        'resample':          {'pixel_scale': pixel_scale},
        'skymatch':          sky_step,
        'outlier_detection': {'skip': not outlier_detection},
    }
    if tweakreg:
        steps['tweakreg'] = {
            'snr_threshold': 5, 'expand_refcat': True,
            'abs_refcat': 'GAIADR3',
            'save_catalogs': True, 'save_results': True,
            'output_dir': stage3,
        }
    else:
        steps['tweakreg'] = {'skip': True}

    print(f'Stage 3  ({len(cal_files)} files, tweakreg={"ON" if tweakreg else "OFF"}, '
          f'skymatch={"match" if skymatch else "OFF"}'
          f'{" subtract="+str(sky_subtract) if skymatch else ""}, '
          f'outlier={"ON" if outlier_detection else "OFF"}) ...')
    calwebb_image3.Image3Pipeline.call(
        asn_file, steps=steps, output_dir=stage3, save_results=True)
    out = sorted(glob.glob(os.path.join(stage3, '*_i2d.fits')))
    print(f'{len(out)} i2d file(s)')
    return out


# ═══════════════════════════════════════════════════════════════════════
#  Mosaic helpers (like MIRI pipeline_utils)
# ═══════════════════════════════════════════════════════════════════════

def load_mosaic(i2d_path):
    """Load i2d, mask zeros, return float array."""
    with fits.open(i2d_path) as hdul:
        sci = hdul['SCI'].data.astype(float)
    sci[sci == 0] = np.nan
    return sci


def get_stats(sci):
    """Return (median, std, neg%) from sigma-clipped stats."""
    _, med, std = sigma_clipped_stats(sci[np.isfinite(sci)], sigma=3)
    neg = 100 * np.nansum(sci < 0) / np.sum(np.isfinite(sci))
    return med, std, neg


def stats_label(med, std, neg):
    """Format stats for plot annotation."""
    return f'med={med:.4f}  σ={std:.4f}  neg={neg:.1f}%'


def plot_mosaic(sci, title, filename, fig_dir='.', cmap='afmhot', dpi=150):
    """Plot mosaic with stats annotation, transparent NaN, matched colorbar."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    med, std, neg = get_stats(sci)
    fig, ax = plt.subplots(figsize=(14, 12), facecolor='none')
    norm = simple_norm(sci, 'sqrt', percent=99.5)
    cm = plt.get_cmap(cmap).copy(); cm.set_bad('none')
    im = ax.imshow(sci, cmap=cm, origin='lower', norm=norm)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.text(0.03, 0.97, stats_label(med, std, neg),
            transform=ax.transAxes, fontsize=12, va='top', color='white',
            family='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.75))
    ax.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3%', pad=0.1)
    plt.colorbar(im, cax=cax, label='MJy/sr')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, filename), dpi=dpi,
                bbox_inches='tight', transparent=True)
    plt.show()
    return med, std, neg


def plot_comparison(data1, title1, data2, title2, filename, fig_dir='.',
                    cmap='afmhot', dpi=150):
    """Side-by-side comparison of two mosaics."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10), facecolor='none')
    cm = plt.get_cmap(cmap).copy(); cm.set_bad('none')
    for ax, data, title in [(ax1, data1, title1), (ax2, data2, title2)]:
        med, std, neg = get_stats(data)
        norm = simple_norm(data, 'sqrt', percent=99.5)
        im = ax.imshow(data, cmap=cm, origin='lower', norm=norm)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.text(0.03, 0.97, stats_label(med, std, neg),
                transform=ax.transAxes, fontsize=11, va='top', color='white',
                family='monospace',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.75))
        ax.axis('off')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='3%', pad=0.1)
        plt.colorbar(im, cax=cax, label='MJy/sr')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, filename), dpi=dpi,
                bbox_inches='tight', transparent=True)
    plt.show()

    s_ours = get_stats(data1)[1]
    s_mast = get_stats(data2)[1]
    print(f'Our σ:  {s_ours:.4f}  |  MAST σ: {s_mast:.4f}  |  '
          f'Improvement: {(1 - s_ours/s_mast)*100:.1f}%')


def plot_gaia_overlay(stage3, filt, pixel_scale, fig_dir='.', dpi=150):
    """Gaia DR3 overlay zoomed on SMC-SW-Bar-3 (nrcb side).
    Only shows sources where the mosaic has valid data."""
    from astropy.wcs import WCS
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from astroquery.gaia import Gaia

    i2d = sorted(glob.glob(os.path.join(stage3, '*_i2d.fits')))[0]
    with fits.open(i2d) as hdul:
        sci = hdul['SCI'].data.astype(float)
        wcs = WCS(hdul['SCI'].header)
    sci[sci == 0] = np.nan
    ny, nx = sci.shape

    center = wcs.pixel_to_world(nx // 2, ny // 2)
    radius = (nx * pixel_scale / 3600) * u.deg
    Gaia.ROW_LIMIT = 2000
    job = Gaia.cone_search_async(
        coordinate=center, radius=radius,
        table_name='gaiadr3.gaia_source',
        columns=['source_id', 'ra', 'dec', 'phot_g_mean_mag'])
    gaia = job.get_results()
    gaia = gaia[gaia['phot_g_mean_mag'] < 20]
    coords = SkyCoord(gaia['ra'], gaia['dec'], unit='deg')
    gx, gy = wcs.world_to_pixel(coords)

    # Filter: nrcb side (right half) + inside mosaic bounds
    in_bar3 = (gx > nx // 2) & (gx < nx) & (gy > 0) & (gy < ny)
    gx_b, gy_b = gx[in_bar3], gy[in_bar3]
    gmag_b = gaia['phot_g_mean_mag'][in_bar3]

    # Filter: only keep sources where mosaic has valid data (not NaN)
    has_data = np.array([np.isfinite(sci[int(np.clip(y, 0, ny-1)),
                                         int(np.clip(x, 0, nx-1))])
                         for x, y in zip(gx_b, gy_b)])
    gx_b, gy_b, gmag_b = gx_b[has_data], gy_b[has_data], gmag_b[has_data]

    zoom = sci[:, nx//2:]
    fig, ax = plt.subplots(figsize=(12, 14), facecolor='none')
    norm = simple_norm(zoom, 'sqrt', percent=99.5)
    cm = plt.get_cmap('Greys_r').copy(); cm.set_bad('none')
    ax.imshow(zoom, cmap=cm, origin='lower', norm=norm)
    ax.scatter(gx_b - nx//2, gy_b, s=80, facecolors='none',
               edgecolors='lime', linewidths=1.2, label='Gaia DR3')
    for i in range(len(gx_b)):
        if gmag_b[i] < 16:
            ax.annotate(f'G={gmag_b[i]:.1f}',
                        xy=(gx_b[i] - nx//2, gy_b[i]),
                        xytext=(6, 6), textcoords='offset points',
                        color='lime', fontsize=7, alpha=0.9)
    ax.legend(fontsize=11, loc='upper right')
    ax.set_title(f'{filt} — SMC-SW-Bar-3  |  Gaia DR3  |  '
                 f'{len(gx_b)} sources\n'
                 f'Circles on stars = good WCS', fontsize=11)
    ax.axis('off'); plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'{filt}_gaia_overlay_SMCbar3.png'),
                dpi=dpi, bbox_inches='tight', transparent=True)
    plt.show()
    print(f'{len(gx_b)} Gaia sources on valid pixels in SMC-SW-Bar-3')

def plot_nircam_histogram(data_sub, good_sub, filt, fig_dir='.', dpi=150):
    """Background histogram with Gaussian overlay (counts, not density)."""
    from scipy.stats import norm as gaussnorm

    bg_pixels = data_sub[good_sub]
    _, mean_bg, pixel_noise = sigma_clipped_stats(bg_pixels, sigma=3)

    fig, ax = plt.subplots(figsize=(10, 6), facecolor='none')

    counts, bins, _ = ax.hist(bg_pixels, bins=300, color='steelblue',
                              edgecolor='none', alpha=0.8,
                              label='Background pixels')

    x = np.linspace(-4*pixel_noise, 4*pixel_noise, 500)
    bin_width = bins[1] - bins[0]
    gauss_scaled = gaussnorm.pdf(x, loc=mean_bg, scale=pixel_noise) * len(bg_pixels) * bin_width
    ax.plot(x, gauss_scaled, 'r-', lw=2, label=f'Gaussian (σ={pixel_noise:.3f})')

    for ns, alpha in [(1, 0.3), (2, 0.15), (3, 0.07)]:
        ax.axvspan(mean_bg - ns*pixel_noise, mean_bg + ns*pixel_noise,
                   color='gray', alpha=alpha)

    ax.axvline(0, color='red', ls='--', lw=1.5, label='zero')
    ax.axvline(mean_bg, color='orange', ls='-', lw=1.5,
               label=f'mean = {mean_bg:.4f}')

    within_1s = 100 * np.sum(np.abs(bg_pixels - mean_bg) < pixel_noise) / len(bg_pixels)
    within_2s = 100 * np.sum(np.abs(bg_pixels - mean_bg) < 2*pixel_noise) / len(bg_pixels)
    within_3s = 100 * np.sum(np.abs(bg_pixels - mean_bg) < 3*pixel_noise) / len(bg_pixels)
    n_neg = 100 * np.sum(bg_pixels < 0) / len(bg_pixels)

    stats_text = (f'σ = {pixel_noise:.4f} MJy/sr\n'
                  f'mean = {mean_bg:.4f}\n'
                  f'within 1σ: {within_1s:.1f}%\n'
                  f'within 2σ: {within_2s:.1f}%\n'
                  f'within 3σ: {within_3s:.1f}%\n'
                  f'negative: {n_neg:.1f}%')
    ax.text(0.97, 0.95, stats_text, transform=ax.transAxes,
            fontsize=11, va='top', ha='right', family='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9))

    ax.set_xlabel('Pixel Value (MJy/sr)', fontsize=12)
    ax.set_ylabel('N pixels', fontsize=12)
    ax.set_title(f'{filt} — Background after sky subtraction', fontsize=14)
    ax.set_xlim(mean_bg - 4*pixel_noise, mean_bg + 4*pixel_noise)
    ax.legend(loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'{filt}_background_histogram.png'),
                dpi=dpi, bbox_inches='tight', transparent=True)
    plt.show()

    print(f'σ={pixel_noise:.4f}  |  1σ: {within_1s:.1f}%  2σ: {within_2s:.1f}%  3σ: {within_3s:.1f}%')

# ═══════════════════════════════════════════════════════════════════════
#  Summary
# ═══════════════════════════════════════════════════════════════════════

def print_summary(filt, cfg, m_ours=None,
                  clean_1f=True, tweakreg=True, skymatch=False,
                  sky_subtract=False, outlier_det=True):
    """Print final pipeline summary."""
    print('=' * 60)
    print(f'  {filt} NIRCam — SMC-SW-Bar-3 (GO-5952)')
    print('=' * 60)
    print(f'  Type:        {"SW narrow" if cfg["is_narrow"] else "SW broad" if cfg["is_short"] else "LW broad"}')
    print(f'  Pixel scale: {cfg["pixel_scale"]}"/px')
    print(f'  1/f corr:    {"ON" if clean_1f else "OFF"}')
    print(f'  tweakreg:    {"Gaia DR3" if tweakreg else "OFF"}')
    print(f'  skymatch:    {"match (subtract="+str(sky_subtract)+")" if skymatch else "OFF"}')
    print(f'  outlier det: {"ON" if outlier_det else "OFF"}')
    if cfg['is_narrow']:
        print(f'  equalization: ON (scalar per detector)')
    if m_ours:
        med, std, neg = m_ours
        print(f'  Result:      med={med:.4f}  σ={std:.4f}  neg={neg:.1f}%')
    print('=' * 60)
