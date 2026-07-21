import scipy.ndimage
import numpy as np

def circularize(psf):
    """
    This function takes an array corresponding to a PSF and circularizes it by rotating the array and averaging it with the previous 
    iteration of the array.

    param psf: the two-dimensional array corresponding to the PSF that is to be circularized
    """
    cntr = 14
    while (cntr > 0):
        theta = (360./2**cntr)
        # print('theta,', theta)
        g_rot1 = scipy.ndimage.rotate(psf,theta,reshape=False,order=1)
        psf = 0.5*(psf+g_rot1)
        cntr = cntr - 1
    return psf

def cent_circ(xcen, ycen, psf):
    circ = np.zeros_like(psf)
    
    a = psf.shape[0]
    b = psf.shape[1]
    
    [X, Y] = np.meshgrid(np.arange(b) - xcen, np.arange(a) - ycen)
    R = np.sqrt(np.square(X) + np.square(Y))
    rad = np.arange(1, np.max(R), 1)
    bin_size = 1
    
    for i in rad:
        mask = (np.greater(R, i - bin_size)) & np.less(R, i + bin_size)
        values = psf[mask]
        circ[mask] = np.mean(values)
    
    circ[xcen, ycen] = psf[xcen,ycen]
    return circ
    
     