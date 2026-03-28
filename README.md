# JWST Imaging Pipeline — SMC-SW-Bar-3 (GO-5952)

Custom data reduction for JWST/MIRI imaging of SMC-SW-Bar-3.  
Program GO-5952 (PI: J. Roman-Duval), Cycle 3.  
Proposal: https://www.stsci.edu/jwst/phase2-public/5952.pdf

Last updated: March 2026

## Pipeline

```
Stage 1 (detector1 + fix_rateints)
  → Stage 2 (image2, no resample)
    → [F2100W only: column clean → background subtraction (Clark GO-3429)]
    → WCS shifts (from F560W tweakreg)
    → Lyot flag
    → Stage 3 (image3, drizzle)
    → 2D polynomial sky subtraction
```

**Key customizations vs MAST:**
- `ipc: skip` — current ref files add noise (K. Gordon)
- `jump.rejection_threshold: 5.0σ` — less aggressive for extended emission
- `fix_rateints_to_rate` — re-average integrations with NaN handling (K. Gordon, `miri_clean.py`)
- WCS: two-pass tweakreg on F560W, shifts reused for all other filters
- F2100W: column cleaning + dedicated background field from C. Clark (GO-3429)

## Repository structure

```
├── README.md
├── pipeline_utils.py          # Shared pipeline functions
├── skysub_utils.py            # 2D polynomial sky subtraction functions
├── environment.yml
├── miri/
│   ├── F560W_pipeline.ipynb   # Reference filter (WCS alignment measured here)
│   ├── F560W_skysub.ipynb
│   ├── F770W_pipeline.ipynb
│   ├── F770W_skysub.ipynb
│   ├── F1000W_pipeline.ipynb
│   ├── F1000W_skysub.ipynb
│   ├── F1130W_pipeline.ipynb
│   ├── F1130W_skysub.ipynb
│   ├── F1500W_pipeline.ipynb
│   ├── F1500W_skysub.ipynb
│   ├── F2100W_pipeline.ipynb  # Includes column clean + Clark background
│   └── F2100W_skysub.ipynb
└── nircam/                    # (to be added)
```

## Setup

1. Clone: `git clone https://github.com/melyaj/SMC_GO5952.git`
2. Environment: `conda env create -f environment.yml && conda activate jwst`
3. Each notebook has a **Configuration** cell at the top — set `BASE_DIR` and CRDS paths there.
4. Download uncal files from [MAST](https://mast.stsci.edu/) (program GO-5952). Data is not included in this repo.

## Dependencies

Python 3.11+, jwst ≥1.20, astropy, numpy, scipy, matplotlib, tweakwcs. See `environment.yml`.

## Authors

Meriem Elyajouri (STScI)

## Acknowledgments

Karl Gordon — column cleaning algorithm and pipeline guidance  
Chris Clark — dedicated background observations (GO-3429)  
Liz Tarantino — pipeline resources and references
