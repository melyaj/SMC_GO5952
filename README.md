# JWST Imaging Pipeline — SMC-SW-Bar-3 (GO-5952)

Custom data reduction for JWST/MIRI and JWST/NIRCam imaging of SMC-SW-Bar-3.  
Program GO-5952 (PI: J. Roman-Duval), Cycle 3.  
Proposal: https://www.stsci.edu/jwst/phase2-public/5952.pdf

Last updated: March 2026

## MIRI Pipeline (6 filters)

Filters: F560W (5.6 µm), F770W (7.7 µm), F1000W (10.0 µm), F1130W (11.3 µm), F1500W (15.0 µm), F2100W (21.0 µm)  
Pixel scale: 0.11″/px

```
Stage 1 (detector1 + fix_rateints)
  → Stage 2 (image2, no resample)
    → Lyot coronagraph flagging (rows ≥ 700, cols < 310)
    → [F2100W: column clean (Karl Gordon) → bkgsub (Clark GO-3429)]
    → WCS shifts (from F560W tweakreg, reused for all filters)
    → Stage 3 (image3, tweakreg OFF, skymatch match subtract=False)
    → 2D polynomial sky subtraction(filter_skysub.ipynb)
```

Key customizations vs MAST:
- `ipc: skip` — current reference files add noise (K. Gordon)
- `jump.rejection_threshold: 5.0σ` — less aggressive for extended emission
- `fix_rateints_to_rate` — re-average integrations with NaN handling (K. Gordon, `miri_clean.py`)
- WCS: two-pass tweakreg on F560W, shifts applied to all other filters
- F2100W: column cleaning before background subtraction + dedicated background from C. Clark (GO-3429)

## NIRCam Pipeline (8 filters)

LW broadband: F300M (3.0 µm), F335M (3.35 µm), F360M (3.6 µm), F444W (4.4 µm) — 0.042″/px, 2 detectors  
SW broadband: F150W (1.5 µm), F200W (2.0 µm) — 0.021″/px, 8 detectors  
SW narrow-band: F187N (1.87 µm, Paschen-α), F212N (2.12 µm, H₂ 1-0 S(1)) — 0.021″/px, 8 detectors

```
Stage 1 (detector1, snowball flagging, 1/f OFF)
  → Stage 2 (image2, no resample)
    → [F187N/F212N: per-detector equalization (refA method)]
    → Stage 3 (image3, tweakreg Gaia DR3, skymatch match subtract=False)
    → 2D polynomial sky subtraction (filter_skysub.ipynb)
```

Key customizations vs MAST:
- `suppress_one_group: False` — retains pixels with one good group above saturation
- `expand_large_events: True` — flags snowball halos around cosmic rays
- `clean_flicker_noise: OFF` — not applied; 1/f noise is weak in this dataset
- `tweakreg: Gaia DR3` — absolute astrometric alignment for all filters (including narrow-band)
- `skymatch: match, subtract=False` for broadband — matches inter-detector levels without subtracting signal; `OFF` for narrow-band (emission fills the field)
- `outlier_detection: ON` for broadband, `OFF` for narrow-band (level residuals cause false flags)

### Narrow-band equalization (F187N, F212N)

With skymatch disabled, each SW detector retains its own instrumental background level, creating visible seams in the mosaic. Standard equalization methods fail for narrow-band filters because the diffuse line emission (Paschen-α, H₂) biases the per-detector median measurement.

Solution — **module A reference equalization** (`equalize_detectors_refA`):  
Module A detectors (nrca1–nrca4) observe the SMC stellar field without significant line emission. For each dither position, the background level is measured on module A only (sigma-clipped median), then applied to the corresponding module B detectors (nrcb1–nrcb4, which contain the extended SMC emission). Module B never measures its own background, it inherits the level from module A at the same dither. This avoids over-subtraction of the Pa-α / H₂ emission.

Note: tweakreg (Gaia DR3) was enabled for all filters including narrow-band.

### Sky subtraction

All filters: 2D polynomial (degree 2) fitted to background pixels after masking sources (segmentation map + sigma clipping + dilation). Narrow-band filters use more aggressive masking parameters (σ=1.5, dilation=15px) to exclude diffuse line emission.

## Repository structure

```
├── README.md
├── pipeline_utils.py              # MIRI pipeline functions
├── nircam_pipeline_utils.py       # NIRCam pipeline functions
├── skysub_utils.py                # 2D polynomial sky subtraction (shared)
├── environment.yml
├── miri/
│   ├── F560W_pipeline.ipynb       # Reference filter (WCS measured here)
│   ├── F560W_skysub.ipynb
│   ├── F770W_pipeline.ipynb
│   ├── F770W_skysub.ipynb
│   ├── F1000W_pipeline.ipynb
│   ├── F1000W_skysub.ipynb
│   ├── F1130W_pipeline.ipynb
│   ├── F1130W_skysub.ipynb
│   ├── F1500W_pipeline.ipynb
│   ├── F1500W_skysub.ipynb
│   ├── F2100W_pipeline.ipynb      # Column clean + Clark background
│   └── F2100W_skysub.ipynb
└── nircam/
    ├── F300M_pipeline.ipynb
    ├── F300M_skysub.ipynb
    ├── F335M_pipeline.ipynb
    ├── F335M_skysub.ipynb
    ├── F360M_pipeline.ipynb
    ├── F360M_skysub.ipynb
    ├── F444W_pipeline.ipynb
    ├── F444W_skysub.ipynb
    ├── F150W_pipeline.ipynb
    ├── F150W_skysub.ipynb
    ├── F200W_pipeline.ipynb
    ├── F200W_skysub.ipynb
    ├── F187N_pipeline.ipynb       # refA equalization
    ├── F187N_skysub.ipynb
    ├── F212N_pipeline.ipynb       # refA equalization
    └── F212N_skysub.ipynb
```

Each notebook has a **Configuration** cell with run flags (`run_detector1`, `run_image2`, etc.) to toggle individual pipeline stages on/off.

## Setup

1. Clone: `git clone https://github.com/melyaj/SMC_GO5952.git`
2. Environment: `conda env create -f environment.yml && conda activate jwst`
3. Set `BASE_DIR` and CRDS paths in the Configuration cell of each notebook.
4. Download uncal files from [MAST](https://mast.stsci.edu/) (program GO-5952). Data is not included in this repo.

## Dependencies

Python 3.11+, jwst ≥ 1.20, astropy, numpy, scipy, matplotlib, photutils, reproject, tweakwcs, astroquery. See `environment.yml`.

## Contributors

Meriem Elyajouri (STScI)

## Acknowledgments

## Acknowledgments

Karl Gordon (STScI) — NIRCam and MIRI pipeline guidance; parts of the MIRI reduction code are adapted from his work:  
https://github.com/STScI-MIRI/Imaging_ExampleNB/tree/main

Liz Tarantino — NIRCam and MIRI pipeline resources and references; parts of the reduction code are adapted from her work:  
https://github.com/liztino/sextansA_PAH_paper/tree/main/data_reduction

Chris Clark — background subtraction methodology and dedicated background observations for F2100W (GO-3429; M101)