# SMC GO-5952 — Data provenance
One line per processing transition: which code produces which files
from which inputs. Code: https://github.com/melyaj/SMC_GO5952
(everything below lives in this repo). Data levels refer to the Box
backup folders (`SMC_GO5952_backup/`).

```
MAST raw (GO-5952)
   │  [A] per-filter reduction notebooks
   ▼
01_stage3/                      {miri,nircam}_{FILT}_final_i2d.fits
   │  [B] sky subtraction (MIRI)
   ▼
02_stage3_skysub/               *_skysub.fits (MIRI only; NIRCam unchanged)
   │  [C] zero-point constants (all 14)
   ▼
03_stage3_zeropoint/            *_skysub_zp.fits (MIRI), *_zp.fits (NIRCam)
   │  [D] PSF matching + reprojection
   ▼
04_psf_matched_science_ready/   {FILT}_matchedF2100W.fits
   │  [E] continuum subtraction & analysis (preliminary)
   ▼
derived_preliminary/            pah/, science/
```

## [A] MAST raw → 01_stage3  (one notebook per filter)

| Filter(s) | Notebook | Notes |
|---|---|---|
| NIRCam F150W…F444W (8) | `nircam/{FILT}_pipeline.ipynb` | stage 1→3; tweakreg on Gaia DR3; broadband: skymatch (match, subtract=False); narrowband (F187N, F212N): skymatch OFF, module-A→B level equalization (protects Pa-α / H₂ diffuse emission) |
| MIRI F560W…F1500W (5) | `miri/{FILT}_pipeline.ipynb` | stage 1→3; tweakreg OFF; skymatch (match, subtract=False) |
| MIRI F2100W | `miri/F2100W_pipeline.ipynb` | + column cleaning (K. Gordon method) and dedicated background subtraction (GO-3429, C. Clark) before stage 3 |

Shared helpers: `pipeline_utils.py`, `nircam_pipeline_utils.py`.

## [B] 01 → 02: MIRI 2D sky subtraction (one notebook per filter)

| Filter(s) | Notebook | Output |
|---|---|---|
| MIRI ×6 | `miri/{FILT}_skysub.ipynb` (uses `skysub_utils.py`) | `*_i2d_skysub.fits` |

Degree-2 2D polynomial fitted to background pixels (segmentation mask
+ 2σ clip + 10 px dilation + 50 px edge crop). The fitted model and
mask are stored in the SKYMODEL / SKYMASK extensions (reversible).
NIRCam: no post-mosaic sky subtraction (see [A]).

## [C] 02 → 03: zero-point constants (all 14 filters, one script)

| Code | Input | Output |
|---|---|---|
| `psf_match/00_subtract_zeropoint.py` + `psf_match/zeropoint_offsets.ecsv` | MIRI `*_i2d_skysub.fits`, NIRCam `*_i2d.fits` | `*_zp.fits` (SCI − ZPOFF; ZPOFF/ZPSYS in headers) |

Astrometric registration: after PSF matching, all bands are registered
to the Gaia (NIRCam) frame. Final solution = GLOBAL weighted
least-squares over the 87 pairwise differential offsets
(psf_match/13_global_registration.py; S/N>6 stars per pair; NIRCam mean
anchored to Gaia; supersedes the incremental steps 11/12). Two
methodological rules: measure offsets only with stars of high S/N in
the band being measured (faint-star centroids regress toward the
reference position, diluting offsets up to x3), and close the system
globally (adjacent-band differentials are ~3x more precise than
band-vs-reference). Corrections = CRVAL updates (no resampling),
cumulative ASTCORR keywords in headers, tables
astrometric_{offsets,offsets_v2,global}.ecsv. Final state: all 14
bands within 2.4 mas (F1500W 4.4), pair residuals <= 1.6 sigma.

Offsets = 3σ-clipped medians in a dark reference cavity common to all
14 bands (measured on the matched maps; constants commute with
convolution/reprojection — proof: `con_sub/tests/verify_zeropoint_commutes.py`).

## [D] 03 → 04: PSF matching to F2100W + common grid (scripts, run in order)

| Step | Script | Role |
|---|---|---|
| 1 | `psf_match/01_make_psfs.py` | STPSF 2.2.0 PSFs per filter, mosaic orientation |
| 2 | `psf_match/02_make_kernels.py` | Aniano+2011 kernels {FILT}→F2100W (κ=0.9 for F187N/F212N/F300M/F360M/F1130W, else 1.0) |
| 3 | `psf_match/03_convolve_images.py` | convolve SCI; ERR = √(ERR²⊗k²) |
| 4 | `psf_match/04_reproject_common.py` | reproject to 0.11″/px north-up 1244×1244 grid |
| 5–7 | `psf_match/05…07_*.py` | validation: spike rotation, κ scan, matched FWHM |

Configuration (paths, filters, grid): `psf_match/config.py`.

## [E] 04 → derived (preliminary analysis, scripts in `con_sub/`, run in order)

| Step | Script | Output |
|---|---|---|
| 0 | `con_sub/00_zeropoint.py` | verification only: dark-cavity median ≈ 0 in all bands |
| 1 | `con_sub/01_consub_kmethod.py` | PAH 3.3/7.7/11.3 maps (k-method, Tarantino+2025) |
| 2 | `con_sub/02_band_ratios.py` | band-ratio maps (SNR>3) |
| 3, 5 | `con_sub/03_d21_ratio_plane.py`, `05_d21_grid_v2.py` | Draine+2021 diagnostic plane |
| 4, 4b | `con_sub/04_consub_alternatives.py`, `04b_donnelly.py` | method comparisons |
| 6 | `con_sub/06_science_quicklook.py` | Σ_PAH, R_PAH, Pa-α, H₂ quicklooks |

Verification suite: `con_sub/tests/` (k-method algebra & Monte-Carlo,
zero-point commutation, canonical-order equivalence).
