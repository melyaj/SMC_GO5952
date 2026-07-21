"""
psfmatch_utils.py — Shared helpers for the PSF matching pipeline
================================================================
SMC GO-5952. Aniano+ 2011 method, adapted from Liz Tarantino's
sextansA_PAH_paper psf-match code (psf_match.py, convolve_image.py).

Functions:
    grid_pa_y        — position angle (E of N) of the +y axis of a mosaic grid
    psf_rot_angle    — rotation to bring a detector-frame PSF onto a mosaic grid
    resize_center    — trim/pad a PSF array to a given grid size (Tarantino resize)
    aniano_kernel    — Aniano+ 2011 kernel: IFFT( FFT(psf_target)/FFT(psf_source) x filter )
    kernel_to_image_scale — resample a kernel to an image pixel scale (odd size, sum=1)
    convolve_with_kernel  — NaN-safe overlap-add convolution with coverage normalization
    measure_spike_angle   — orientation of diffraction spikes (m=6 azimuthal phase)
"""
import numpy as np
import scipy.fftpack as fp
from scipy.signal import oaconvolve
from astropy.wcs import WCS
import astropy.units as u

from congrid import congrid


# ---------------------------------------------------------------- geometry
def grid_pa_y(sci_header):
    """Position angle (deg E of N) of the +y pixel axis of a mosaic, at the
    image center. north-up grid -> 0."""
    wcs = WCS(sci_header)
    x0, y0 = sci_header["NAXIS1"] / 2, sci_header["NAXIS2"] / 2
    c0 = wcs.pixel_to_world(x0, y0)
    c1 = wcs.pixel_to_world(x0, y0 + 50)
    return c0.position_angle(c1).to_value(u.deg)


def psf_rot_angle(sci_header):
    """Angle (deg) to pass to scipy.ndimage.rotate to bring a detector-frame
    STPSF PSF onto this mosaic's pixel grid.

    Tarantino rotates by -PA_APER for a north-up grid; generalized here to
    grid_pa - PA_APER for an arbitrarily rotated drizzle grid.
    Validate the sign with 05_validate_rotation.py (star diffraction spikes).
    """
    return grid_pa_y(sci_header) - sci_header["PA_APER"]


# ---------------------------------------------------------------- PSF grids
def resize_center(img, grid_size):
    """Trim (around the peak) or zero-pad a PSF to grid_size (Tarantino resize)."""
    from center import center

    n = np.shape(img)[0]
    if n > grid_size:
        xcen, ycen = center(img)
        ind = int(grid_size / 2)
        return img[(xcen - ind):(xcen + ind + 1), (ycen - ind):(ycen + ind + 1)]
    elif n < grid_size:
        new = np.zeros((grid_size, grid_size), dtype=float)
        start = int((grid_size - n) / 2.0) + 1
        new[start:start + n, start:start + n] = img
        return new
    return img


# ---------------------------------------------------------------- kernel
def aniano_filter(n, pix_size, fwhm1, kappa=1.0):
    """Low-pass cosine filter of Aniano+ 2011 (vectorized version of the
    frequency loop in Tarantino psf_match.py).

    n        : grid size (pixels)
    pix_size : pixel scale (arcsec)
    fwhm1    : FWHM of the source PSF (arcsec)
    """
    k = fp.fftfreq(n, pix_size) * 2 * np.pi
    kx, ky = np.meshgrid(k, k, indexing="ij")
    freq = np.hypot(kx, ky)

    k_h = kappa * 2 * np.pi / fwhm1
    k_l = 0.7 * k_h

    filt = np.zeros_like(freq)
    filt[freq <= k_l] = 1.0
    mid = (freq > k_l) & (freq < k_h)
    filt[mid] = 0.5 * (1 + np.cos(np.pi * (freq[mid] - k_l) / (k_h - k_l)))
    return filt


def aniano_kernel(psf1, psf2, pix_size, fwhm1, kappa=1.0):
    """Aniano+ 2011 convolution kernel psf1 -> psf2.

    Both PSFs must be on the same grid (same size, same pixel scale, centered)
    and normalized to sum 1. Returns (kernel, diagnostics dict).
    D  = sum |psf1 (x) kernel - psf2|   (Aniano accuracy metric)
    W- = sum of negative kernel values  (Aniano negative weight)
    """
    from astropy.convolution import convolve_fft

    fft_psf1 = fp.fft2(psf1)
    fft_psf2 = fp.fft2(psf2)

    filt = aniano_filter(len(psf1), pix_size, fwhm1, kappa=kappa)

    ker = fp.ifft2((fft_psf2 / fft_psf1) * filt)
    ker = fp.fftshift(ker.real)
    ker = ker / np.sum(ker)

    neg_sum = np.sum(ker[ker < 0])
    conv = convolve_fft(psf1, ker, boundary="wrap", allow_huge=True)
    D = np.sum(np.abs(conv - psf2))

    return ker, {"D": D, "W-": neg_sum, "kappa": kappa, "conv_check": conv}


def kernel_to_image_scale(ker, pix_ker, pix_img):
    """Resample a kernel to an image pixel scale (spline, odd output size,
    renormalized to sum 1).

    Resampling the (small) kernel instead of the (huge) image — the reverse of
    Tarantino convolve_image.py — is equivalent and fits in memory for the
    NIRCam SW mosaics.
    """
    size_new = int(round(len(ker) * pix_ker / pix_img))
    if size_new % 2 == 0:
        size_new += 1
    new = congrid(ker, np.array([size_new, size_new]), method="spline",
                  centre=True, minusone=True)
    return new / np.sum(new)


# ---------------------------------------------------------------- convolution
def convolve_with_kernel(data, ker, normalize_coverage=True):
    """NaN-safe convolution (overlap-add FFT, memory-bounded).

    NaNs are zero-filled before convolution and restored after (Tarantino
    convolve_image.py). With normalize_coverage=True the result is divided by
    the convolved coverage mask, which removes the edge dimming next to
    NaN regions (Lyot cut, mosaic borders).
    """
    valid = np.isfinite(data)
    filled = np.where(valid, data, 0.0)

    conv = oaconvolve(filled, ker, mode="same")
    if normalize_coverage:
        cov = oaconvolve(valid.astype(float), ker, mode="same")
        with np.errstate(invalid="ignore", divide="ignore"):
            conv = np.where(cov > 0.5, conv / cov, np.nan)
    conv[~valid] = np.nan
    return conv


def convolve_variance(var, ker):
    """Propagate a variance map through the convolution.

    For a linear convolution y = sum(k_i x_i), the exact (correlation-free)
    propagation is var_y = sum(k_i^2 var_i) = var (x) ker^2. Unlike the
    var (x) ker approximation, ker^2 is non-negative, so a positive variance
    map can never convolve to negative values (the kernels have negative
    Aniano lobes W-). The drizzled noise is already pixel-correlated, so this
    remains an approximation — see README."""
    kk = ker ** 2
    s2 = np.sum(kk)
    out = convolve_with_kernel(var, kk / s2) * s2
    return np.clip(out, 0.0, None)


# ---------------------------------------------------------------- validation
def measure_spike_angle(img, r_in=8, r_out=40, m=6, recenter=False):
    """Orientation (deg, CCW from +y, mod 360/m) of the diffraction spikes,
    from the phase of the m-th azimuthal Fourier component in an annulus.

    Works on a star cutout or on a PSF; comparing the two gives the residual
    rotation error of the PSF model. By default the source is assumed to sit
    at the array center (a nanargmax recenter would lock onto a brighter
    neighbor in crowded fields); pass recenter=True for a PSF array whose
    peak may be slightly off-center."""
    ny, nx = img.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    if recenter:
        cy, cx = np.unravel_index(np.nanargmax(img), img.shape)
    else:
        cy, cx = ny // 2, nx // 2
    dx, dy = xx - cx, yy - cy
    r = np.hypot(dx, dy)
    theta = np.arctan2(dx, dy)  # angle from +y axis

    ann = (r >= r_in) & (r <= r_out) & np.isfinite(img)
    w = img[ann]
    t = theta[ann]
    c = np.sum(w * np.exp(1j * m * t))
    return np.degrees(np.angle(c)) / m % (360.0 / m)
