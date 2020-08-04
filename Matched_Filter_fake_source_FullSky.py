import matplotlib.pyplot as pl
import numpy as np
import healpy as hp
from astropy.convolution import convolve, Gaussian2DKernel


#sigma = 30.0/2.355 / 0.5
#kernel = Gaussian2DKernel(sigma)
#b_area = np.sum(kernel.array / np.max(kernel.array) )*np.radians(0.5/60.0)**2 

#print(b_area)

#OmegaB = (np.pi / 4 / np.log(2)) * np.radians(30.0/60)**2
#print(OmegaB)


nside = 1024
# soild angle of a single pixel
Omega_p = 4 * np.pi / (12*nside**2)

ell_arr = np.arange(3*nside)
# This is the noise spectra, which is an exponential term plus a white noise term
Nell = np.exp( -ell_arr/50.0 ) + 1.0

# create a noise realization
noise_map = hp.synfast(Nell,nside,pol=False)

signal_map = np.zeros(12*nside**2)
# put a 200 mJy point source in pixel 123456
pixel = 123456
signal_map[pixel] = 200.0 / Omega_p # mJy/sr

# We will "observe" with a 30 arcmin beam
fwhm = 30.0
# smoothing convolves the map with a gaussian beam
signal_conv_map = hp.smoothing(signal_map,fwhm = np.radians(fwhm/60.0),pol=False)

#print(np.sum(signal_map),np.sum(signal_conv_map))

b_ell = hp.gauss_beam(np.radians(fwhm/60.0),pol=False,lmax=(3*nside-1))

total_map = signal_conv_map + noise_map

# Matched filter, I use the noise spectra as the filter
# these are the normalizations 
norm  = np.sum( b_ell**2 / Nell)

# This normalization won't work for Fullsky, it only works for a flat metric with the Fourier transform. There is a correction term for a spherical metric
norm2 = np.sum(b_ell)
f_ell = norm2 * b_ell / Nell / norm

# I need the harmonic transform of the map
alms_tot = hp.map2alm(total_map,pol=False)
# I multiply the alms with the matched filter
alms_tot = hp.almxfl(alms_tot,f_ell)
# transform back to real space
total_MF_map = hp.alm2map(alms_tot,nside)

#The beam area Omega_B, in sr
OmegaB = (np.pi / 4 / np.log(2)) * np.radians(fwhm/60)**2

total_MF_map = total_MF_map * OmegaB 

# Now total_MF_map is in mJy 

print('The integrated flux of the point source is',total_MF_map[pixel],' mJy')
