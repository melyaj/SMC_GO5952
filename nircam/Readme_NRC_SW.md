# NIRCam SW Broadband Pipeline Notes
## Filters: F150W, F200W
## Notebook: NRC_SW_F[filter]_pipeline.ipynb

One notebook per filter. Change `FILT` in the config cell and run all cells.

The pipeline steps and parameter choices are identical to the LW broadband
pipeline (F300M, F335M, F360M, F444W). The only differences are detector
geometry and pixel scale.

---

### Differences from MAST default pipeline

**Stage 1 — calwebb_detector1**

| Parameter | MAST | This pipeline |
|-----------|------|---------------|
| `suppress_one_group` | True | False |
| `expand_large_events` | False | True |
| `clean_flicker_noise` | skip | enabled |

- `suppress_one_group = False`: retains pixels with only one valid group above
  saturation, recovering signal in bright emission regions.

- `expand_large_events = True`: flags snowball halos around large cosmic-ray
  events. At SW wavelengths (1.5–2.0 µm) the SMC stellar field is dense and
  snowball halos would contaminate point source photometry if left unflagged.

- `clean_flicker_noise = True`: removes 1/f banding noise. In SW NIRCam data
  the 1/f noise appears as horizontal stripes correlated along detector rows.
  The step operates on the ramp data inside Stage 1 using the image1overf
  algorithm (Willott). MAST leaves this step off by default.

**Stage 2 — calwebb_image2**

No changes. `resample` is skipped (drizzling is deferred to Stage 3).

**Stage 3 — calwebb_image3**

| Step | MAST | This pipeline |
|------|------|---------------|
| skymatch | enabled (match) | skipped |
| tweakreg abs_refcat | internal catalog | Gaia DR3 |
| outlier_detection | enabled | enabled |

- `skymatch = skip`: same reason as LW filters. SMC stellar continuum fills
  all 8 SW detectors. Skymatch would subtract real signal.

- `abs_refcat = GAIADR3`: absolute astrometric alignment against Gaia DR3,
  queried automatically from STScI MAST with proper motion corrections.
  At SW wavelengths the SMC stellar population is bright and dense, providing
  many Gaia matches per detector for robust alignment.

- `outlier_detection`: left enabled.

---

### SW detector geometry

The SW channel uses 8 detectors (4 per module, 0.021 arcsec/px):

- nrca1, nrca2, nrca3, nrca4 — module A (SMC stellar field, outskirts of N66)
- nrcb1, nrcb2, nrcb3, nrcb4 — module B (N66/NGC 346 — science target)

Each dither position produces 8 cal files (one per detector), giving 32 input
files to Stage 3 per filter (8 detectors x 4 dithers).

Diagnostic plots show nrca1 vs nrcb1 as representatives of each module.

---

### Output

Final mosaic: `{FILT}/stage3/*_i2d.fits`  
Units: MJy/sr  
Pixel scale: 0.021 arcsec/px
