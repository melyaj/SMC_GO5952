# NIRCam SW Narrow-Band Pipeline Notes
## Filters: F187N (Paschen-alpha, 1.875 um), F212N (H2 1-0 S(1), 2.122 um)
## Notebook: NRC_SW_narrow_F[filter]_pipeline.ipynb

One notebook per filter. These filters require a separate pipeline from the
broadband filters. The reasons and the solutions are documented below.

NOTE: F187N results are still being validated (negative pixel fraction under
investigation). These notes will be updated once the F187N reduction is finalized.
F212N is considered final.

---

### Why a separate pipeline

Two problems make the standard pipeline and the broadband pipeline inapplicable:

1. F187N and F212N are pupil-wheel filters. The bandpass is narrow (~1%),
   so the detected signal is dominated by line emission (Paschen-alpha or H2)
   on top of a low stellar continuum. The signal-to-noise per pixel in a single
   exposure is much lower than for broadband filters.

2. No blank sky exists anywhere in the field. The SMC stellar continuum and
   line emission fill all 8 SW detectors at all dither positions. Any step that
   attempts to measure and subtract a sky background will remove real signal.

---

### Differences from MAST default pipeline

**Stage 1 — calwebb_detector1**

| Parameter | MAST | This pipeline |
|-----------|------|---------------|
| `suppress_one_group` | True | False |
| `expand_large_events` | False | True |
| `clean_flicker_noise` | skip | enabled |

Same reasoning as broadband filters. The 1/f noise is more pronounced in
narrow-band data because exposures are longer (more groups per ramp) and
the science signal per pixel is weaker, making the residual striping
relatively more significant.

Note: `fit_by_channel` is set to True in the notebook. In practice, Stage 1
was already complete before this notebook was finalized, so the auto-parameter
logic of `clean_flicker_noise` was used. The 1/f profile diagnostic confirmed
the correction was clean with no steps at amplifier boundaries (columns 512,
1024, 1536), so no reprocessing was needed.

**Stage 2 — calwebb_image2**

No changes. `resample` skipped.

**Extra step: per-detector level equalization**

Between Stage 2 and Stage 3, a scalar offset is subtracted from each of the
32 cal files (8 detectors x 4 dithers). The offset is:

    offset(file) = sigma_clipped_median(file) - min(sigma_clipped_median, all files)

This brings all detectors to the same relative background level, removing
the inter-detector seams that appear in the mosaic when skymatch is off.

Important: the sigma-clipped median includes both instrumental background
(zodiacal light, detector pedestal) and diffuse SMC emission. The equalization
does not distinguish between them. It is a relative correction only — it
removes detector-to-detector offsets, not an absolute sky background.
The absolute flux scale must be established in the science analysis using
the continuum estimated from adjacent broadband filters (F182M for F187N,
F210M for F212N).

**Stage 3 — calwebb_image3**

| Step | MAST | This pipeline |
|------|------|---------------|
| tweakreg | enabled | skipped |
| skymatch | enabled | skipped |
| outlier_detection | enabled | skipped |

- `tweakreg = skip`: Gaia DR3 is too sparse for reliable alignment of individual
  SW detectors (2048x2048 px, 0.021"/px, ~43" field of view) in a narrow bandpass
  where few stars are detectable. Attempting tweakreg with loose parameters caused
  some exposures to be shifted by several arcminutes, producing a 7333x15529 pixel
  output mosaic. The native JWST WCS, based on the guide star solution, is accurate
  to ~0.1" and sufficient for the science goals.

- `skymatch = skip`: emission fills the field.

- `outlier_detection = skip`: residual level differences between the equalized
  files are small but nonzero. Outlier detection interprets these as source-level
  inconsistencies and flags real pixels as bad, producing rectangular holes in
  the mosaic corresponding to individual detector footprints.

---

### Known limitations

- Wisp stray-light templates from STScI are not available for F187N or F212N
  (only wide and medium SW filters are covered). Detector nrcb4 shows elevated
  and variable background due to wisps. This is partially mitigated by the
  four-point dither pattern, which averages the wisp signal across dither positions.

- The per-detector equalization subtracts a fraction of the diffuse SMC emission
  along with the instrumental pedestal. The correction is the same for all pixels
  within a given exposure (scalar), so spatial emission structure is fully preserved.

---

### Output

Final mosaic: `{FILT}/stage3/*_i2d.fits`  
Units: MJy/sr  
Pixel scale: 0.021 arcsec/px  
Equalized cal files: `{FILT}/stage2_corrected/*_cornercorr_cal.fits`
