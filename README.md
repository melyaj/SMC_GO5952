# JWST Imaging Pipeline — SMC-SW-Bar-3 (GO-5952)

Custom data reduction pipeline for JWST/MIRI and NIRCam imaging of SMC-SW-Bar-3 (PI: J. Roman-Duval), Cycle: 3, Proposal Category: GO https://www.stsci.edu/jwst/phase2-public/5952.pdf


## Pipeline overview

```
Stage 1 (detector1 + fix_rateints)
  → Stage 2 (image2, no resample)
    → Lyot flag
      → Column clean 
        → Background subtraction (Clark GO-3429)
          → WCS shifts (F560W)
            → Stage 3 (image3, drizzle)
              → 2D polynomial background subtraction (see xx.ipynb)
```

### Key steps

- **Column cleaning:** K. Gordon's `cal_column_clean` algorithm removes column-correlated detector noise. Applied to cal files before background subtraction.
- **Background subtraction:** Dedicated background field from C. Clark (GO-3429), median-stacked into a master background frame.
- **WCS correction:** RA/Dec shifts measured from F560W alignment applied via `tweakwcs`.
- **fix_rateints:** improves the default JWST pipeline's averaging of integrations (K. Gordon).

## Repository structure

```
├── README.md
├── pipeline_utils.py            # Shared helper functions (all filters)
├── environment.yml              # Conda environment specification
├── miri/
│   ├── 7_F2100W_pipeline.ipynb  # Full pipeline for F2100W
│   ├── 7_F770W_pipeline.ipynb   # (to be added)
│   ├── 7_F1000W_pipeline.ipynb  # (to be added)
│   ├── 7_F1130W_pipeline.ipynb  # (to be added)
│   ├── 7_F1500W_pipeline.ipynb  # (to be added)
│   └── 09_bkg_polynomial_subtract.ipynb
└── nircam/                      # (to be added)
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/SMC_GO5952_pipeline.git
cd SMC_GO5952_pipeline
```

### 2. Create the conda environment

```bash
conda env create -f environment.yml
conda activate jwst
```

### 3. Configure paths

Each notebook has a **Configuration** cell at the top where you set:
- `BASE_DIR` — path to your local MIRI/NIRCam data directory
- `BKG_BASE` — For F2100W path to the Clark GO-3429 background observations
- CRDS cache path

### 4. Data

The raw JWST data (uncal files) can be downloaded from [MAST](https://mast.stsci.edu/) using program ID **GO-5952**. The background observations are from program **GO-3429** (PI: C. Clark).

Data is **not** included in this repository, only the pipeline code and notebooks.

## Dependencies

- Python 3.11+
- [jwst](https://github.com/spacetelescope/jwst) pipeline (v1.20+)
- astropy
- numpy, scipy, matplotlib
- tweakwcs
- CRDS (with local cache configured)

See `environment.yml` for the full specification.

## Authors

- Meriem Elyajouri (STScI)

## Acknowledgments

- Karl Gordon — pipeline resources and guidance
- Chris Clark — dedicated background observations (GO-3429) and background subtraction strategy
- Liz Tarantino — pipeline resources and references