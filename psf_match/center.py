#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Find center

@author: Eliz
"""

import numpy as np
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve

def center(psf):

    # smooth the PSF with a 5 pixel kernel
    fwhm = 5
    std = fwhm/2.355
    ker = Gaussian2DKernel(x_stddev=std)

    sm_psf = convolve(psf, ker)
    
    m = np.nanmax(sm_psf)
    
    array = np.where(((m - sm_psf)/m) <= 5e-4)
    
    xcen = array[0]
    ycen = array[1]
    
    return xcen[0], ycen[0]
        
    

