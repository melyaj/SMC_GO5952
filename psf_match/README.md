# PSF matching — SMC GO-5952

Homogenization of the 14 NIRCam+MIRI mosaics to the **F2100W PSF**
(FWHM ≈ 0.67″), then reprojection to the common 0.11″/px north-up grid.
Analysis-ready maps land in `SMC_GO5952/analysis_ready/`.

Method: **Aniano et al. 2011** kernel generation, adapted from
**Liz Tarantino's** `sextansA_PAH_paper/psf-match` code
(Tarantino et al. 2025, Sextans A PAH paper). Helpers `center.py`, `fwhm.py`,
`congrid.py`, `radial_data.py`, `regrid.py`, `circularize_laura.py` are copied
verbatim from her repository.

## Workflow

```
01_make_psfs.py         STPSF PSF per filter (setup_sim_to_match_file on the i2d,
                        oversample 3, odd parity, fov 41.25″ MIRI / 30″ NIRCam),
                        rotated to each mosaic grid (grid_pa − PA_APER)
02_make_kernels.py      Aniano kernel {filt} → F2100W (κ = 1), with the F2100W
                        PSF rotated to the SOURCE mosaic grid (MIRI and NIRCam
                        drizzle grids have different position angles).
                        Diagnostics D and W⁻ + radial-profile figures
03_convolve_images.py   convolution at native pixel scale (kernel resampled to
                        the image scale; overlap-add FFT; NaN-safe with
                        coverage normalization). ERR = sqrt(ERR² ⊗ kernel²)
04_reproject_common.py  reproject_adaptive → grid of reprojected/F2100W_final.fits
05_validate_rotation.py empirical check of the PSF rotation on star
                        diffraction spikes (m=6 azimuthal phase)
```

Products (not in git): `SMC_GO5952/psf_match/{psfs,kernels,convolved,figures}`.

## Inputs

- MIRI: `miri/<filt>/stage3/miri_<filt>_final_i2d_skysub.fits`
- NIRCam: `nircam/<filt>/stage3/nircam_<filt>_final_i2d.fits`
  (no polynomial skysub for NIRCam — see main pipeline README)

## Deviations from Tarantino's original code

- the Fourier low-pass filter loop is vectorized (identical maths)
- the **kernel** is resampled to the image pixel scale instead of resampling
  the image to the kernel scale — required for the 7333×15529 NIRCam SW
  mosaics on 16 GB RAM; mathematically equivalent
- convolution uses `scipy.signal.oaconvolve` (memory-bounded) with coverage
  normalization at NaN edges (Lyot cut, mosaic borders)
- ERR propagation uses var ⊗ kernel² (exact correlation-free propagation,
  positive-definite) instead of Tarantino's var ⊗ kernel, whose negative
  Aniano lobes drive smooth variance regions negative → NaN after sqrt.
  Still approximate: drizzled noise is correlated between neighboring pixels
- PSFs are rotated to each **mosaic grid** (not north-up) so the science data
  are only resampled once, at the final reprojection

## Environment

`conda activate jwst` (stpsf 2.2.0, photutils, reproject, astropy).
STPSF data files: `~/data/stpsf-data` (auto-downloaded).
