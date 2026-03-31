# NIRCam LW Broadband Pipeline Notes
## Filters: F300M, F335M, F360M, F444W
## Notebook: NRC_LW_F[filter]_pipeline.ipynb

One notebook per filter. Change `FILT` in the config cell and run all cells.

---

### Differences from MAST default pipeline

**Stage 1 — calwebb_detector1**

| Parameter | MAST | This pipeline |
|-----------|------|---------------|
| `suppress_one_group` | True | False |
| `expand_large_events` | False | True |
| `clean_flicker_noise` | skip | enabled |

- `suppress_one_group = False`: pixels for which only one group is available above
  the saturation threshold are retained rather than discarded. This recovers signal
  in regions of bright nebular emission where early groups saturate.

- `expand_large_events = True`: the jump step flags extended halos around large
  cosmic-ray events (snowballs). Without this, the halos produce ring-shaped
  artifacts in the final mosaic.

- `clean_flicker_noise = True`: removes 1/f (flicker) noise, which appears as
  horizontal banding correlated along detector rows. The MAST pipeline leaves this
  step off by default because no single parameter set is optimal for all programs.
  The built-in step uses the same algorithm as the community tool image1overf
  (Willott), operating on the ramp data inside Stage 1.

**Stage 2 — calwebb_image2**

No changes. `resample` is skipped (drizzling is deferred to Stage 3).

**Stage 3 — calwebb_image3**

| Step | MAST | This pipeline |
|------|------|---------------|
| skymatch | enabled (match) | skipped |
| tweakreg abs_refcat | internal catalog | Gaia DR3 |
| outlier_detection | enabled | enabled |

- `skymatch = skip`: the SMC stellar continuum and diffuse emission fill both LW
  detectors entirely. The skymatch step would measure this emission as sky and
  subtract it, removing real astrophysical signal. Karl Gordon (STScI) recommended
  skipping skymatch for this field.

- `abs_refcat = GAIADR3`: absolute astrometric alignment is performed against
  Gaia DR3, queried automatically from the STScI MAST server with proper motion
  corrections applied to the observation epoch. MAST uses an internal catalog.
  This places the mosaic on the Gaia reference frame.

- `outlier_detection`: left enabled. With the correct input files (one set of
  cal files per filter, not mixed versions), outlier detection performs correctly
  and reduces the background noise in the final mosaic.

---

### Notes on N66/NGC 346

The two LW detectors cover different sky regions:
- nrcalong (module A): SMC stellar field, outskirts of N66
- nrcblong (module B): N66/NGC 346 — primary science target

Diagnostic plots highlight nrcblong in orange throughout the notebook.

---

### Output

Final mosaic: `{FILT}/stage3/*_i2d.fits`  
Units: MJy/sr  
Pixel scale: 0.042 arcsec/px
