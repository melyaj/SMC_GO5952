"""Shared display helper: footprint-aligned rotation + tight crop."""
import numpy as np
from scipy import ndimage

ANG = -15.0     # optimal footprint angle (bounding-box minimization)

def rotcrop(im, bbox=None):
    r = ndimage.rotate(np.nan_to_num(im, nan=0.0), ANG, reshape=True, order=1)
    m = ndimage.rotate(np.isfinite(im).astype(float), ANG, reshape=True, order=1)
    r[m < 0.95] = np.nan
    if bbox is not None:
        y1, y2, x1, x2 = bbox
        r = r[y1:y2, x1:x2]
    return r

def common_bbox(masks_stack):
    mask = np.all(masks_stack, axis=0)
    mask = ndimage.binary_erosion(mask, iterations=12)
    rm = ndimage.rotate(mask.astype(float), ANG, reshape=True, order=1) > 0.9
    rows = np.where(rm.mean(axis=1) > 0.60)[0]
    cols = np.where(rm.mean(axis=0) > 0.60)[0]
    return rows.min(), rows.max(), cols.min(), cols.max()
