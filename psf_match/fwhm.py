#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Calculates the FWHM of a kernel

@author: Eliz
"""

# from astropy.io import fits
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

# import matplotlib.pyplot as plt
from center import center
import scipy.optimize as opt
from astropy.io import ascii
# from circularize_laura import circularize

# filepath = '/Users/Eliz/Documents/UMD/Research/stinytim/test/'

# # psf2_name = 'SIII_19_v3.fits'
# psf2_name = 'SIII_33_v1.fits'

# psf2_hdu = fits.open(filepath + psf2_name)
# psf = psf2_hdu[0].data
# psf_head = psf2_hdu[0].header


# data = psf
# psf = circularize(psf)

def fwhm(psf, psf_head):

    pixscale = psf_head['PIXELSCL']
    
    # pixscale = abs(psf_head['CDELT1'])*60**2
    
    def twoDGau(data, xo, yo, sigma_x, sigma_y, amplitude, offset):
        """Function to fit, returns 2D gaussian function as 1D array"""
        x = data[0]
        y = data[1]
        xo = float(xo)
        yo = float(yo)    
        g = offset + amplitude*np.exp( - (((x-xo)**2)/(2*sigma_x**2) + ((y-yo)**2)/(2*sigma_y**2)))
        return g.ravel()
    
    cen = center(psf)
    cenx = cen[0]
    ceny = cen[1]
    x = np.linspace(0, psf.shape[1], psf.shape[1])
    y = np.linspace(0, psf.shape[0], psf.shape[0])
    x, y = np.meshgrid(x, y)
    #Parameters: xpos, ypos, sigmaX, sigmaY, amp, baseline
    # subtract background and rescale image into [0,1], with floor clipping
    data = (x,y)
    
    initial_guess = (cenx,ceny,10,10,np.max(psf),0)
    # psf = np.array(psf,dtype='float64')
    
    # subtract background and rescale image into [0,1], with floor clipping
    popt, pcov = opt.curve_fit(twoDGau, data, psf.ravel(), p0=initial_guess)
    xcenter, ycenter, sigmaX, sigmaY, amp, offset = popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]
    fwhm_x = np.abs(4*sigmaX*np.sqrt(-0.5*np.log(0.5)))
    fwhm_y = np.abs(4*sigmaY*np.sqrt(-0.5*np.log(0.5)))
    
    xarcsec = fwhm_x * pixscale
    yarcsec = fwhm_y * pixscale

    fwhm = {'x': fwhm_x, 'y': fwhm_y, 'xarcsec': xarcsec, 'yarcsec': yarcsec }
    
    # print(fwhm)
    
    # pixscale = psf_head['PIXSCALX']
    # xcen, ycen = center(psf)
    
    # print(fwhm['x']*pixscale, fwhm['y']*pixscale)
    
    # cross = np.mean(psf, axis = 0)
    
    # plt.figure(1)
    # plt.clf()
    # plt.plot(cross)
    
    # x2 = cenx + fwhm_x/2
    # x1 = ceny - fwhm_x/2
    
    # plt.axhline(y = np.max(cross)/2, c = 'k')
    # plt.axvline(x1, c = 'k')
    # plt.axvline(x2, c = 'k')
    # plt.xlim(150, 250)
    
    # ans = twoDGau(data, xcenter, ycenter, sigmaX, sigmaY, amp, offset)
    # ans = ans.reshape(np.shape(psf)[0], np.shape(psf)[1])
    
    # cross_ans = np.mean(ans, axis = 0)
    
    # plt.plot(cross_ans, c = 'g', ls = '--')

    
    return fwhm

# path = '/Users/Eliz/Documents/UMD/Research/N76_IR_paper/'
# name = 'line_info.txt'

# lines = ascii.read(path + name)

# psf_x = np.zeros(len(lines))
# psf_y = np.zeros(len(lines))


# for i in range(len(lines)):
#     psf_path = '/Users/Eliz/Documents/UMD/Research/stinytim/psfs/'
    
#     if lines['name'][i] == 'NeV_24':
#         psf_file = 'NeV-24_psf'
#     else:
#         psf_file = lines['name'][i] + '-' + str(round(lines['micron'][i])) + '_psf'

#     psf_hdu = fits.open(psf_path + psf_file + '.fits')
#     psf = psf_hdu[0].data
#     psf_head = psf_hdu[0].header

#     fwhm_val = fwhm(psf, psf_head)
    
#     psf_x[i] = fwhm_val['xarcsec']
#     psf_y[i] = fwhm_val['yarcsec']


# lines.add_column(psf_x, name = 'psf_x')
# lines.add_column(psf_y, name = 'psf_y')

# outname = 'line_info_psf'
# ascii.write(lines, path + outname + '.txt', format = 'fixed width')

# # psf_file = 'SiII-35_psf-S3-20'
# psf_file = 'SIV-11_psf'

# psf_path = '/Users/Eliz/Documents/UMD/Research/SMC_PACS/PSFs/PACSPSF_monochromatic_v2.0/'
# psf_file = 'R_60'

# # # open psf files
# psf_hdu = fits.open(psf_path + psf_file + '.fits')
# psf = psf_hdu[0].data
# psf_head = psf_hdu[0].header

# fwhm_val = fwhm(psf, psf_head)

# print(fwhm_val)


    
# psf1_file = 'FeII-18_psf'  
# # psf1_file = 'SIII_19_v3'
# # psf2_file = 'SIII-33_psf'
# # # psf2_file = 'SIII_33_v1'
    
# # # path for the psf files
# psf_path = '/Users/Eliz/Documents/UMD/Research/stinytim/psfs/'

# # # open psf files
# psf1_hdu = fits.open(psf_path + psf1_file + '.fits')
# psf1 = psf1_hdu[0].data
# psf1_head = psf1_hdu[0].header

# # psf2_hdu = fits.open(psf_path + psf2_file + '.fits')
# # psf2 = psf2_hdu[0].data
# # psf2_head = psf2_hdu[0].header

# width1 = fwhm(psf1, psf1_head)
# width2 = fwhm(psf2, psf2_head)