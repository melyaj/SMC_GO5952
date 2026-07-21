#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Regrids an image with the inputs

@author: Eliz
"""
import numpy as np
from congrid import congrid

# pixel scales are all in arcsec
def regrid(img, orig_pix_scale, goal_pix_scale, method = 'spline', center = True):
    # regridding the data
    size_data = len(img)
    size_new = int(round( float(size_data) * orig_pix_scale / goal_pix_scale))
    
    # print(size_new)
        
    new_img = congrid(img, np.array([size_new, size_new]), method = method, centre = center, minusone = True)
    
    return new_img

