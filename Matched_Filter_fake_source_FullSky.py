import matplotlib.pyplot as pl
import numpy as np
import healpy as hp
from astropy.convolution import convolve, Gaussian2DKernel
from astropy.cosmology import Planck15
from astropy import units as u

#sigma = 30.0/2.355 / 0.5
#kernel = Gaussian2DKernel(sigma)
#b_area = np.sum(kernel.array / np.max(kernel.array) )*np.radians(0.5/60.0)**2 
#print(b_area)
#OmegaB = (np.pi / 4 / np.log(2)) * np.radians(30.0/60)**2
#print(OmegaB)

T_cmb = 2.725
c  = 299792458.0
h  = 6.62606957e-34
k  = 1.3806488e-23

nside = 512
npix = 12*nside**2
# soild angle of a single pixel
Omega_p = 4 * np.pi / npix

ell_arr = np.arange(3*nside)

# This is the noise spectra, which is an exponential term plus a white noise term
Nell = 1e-4 * np.exp( -ell_arr/50.0 ) + 1e-6
#Nl = exp(-l/50.) + 1e-3

# create a noise realization
noise_map = hp.synfast(Nell,nside,pol=False)

signal_map = np.zeros((npix)) #*u.mJy/u.sr
# put a 200 mJy point source in 1k random pixels
pixels = np.random.uniform(low=0,high=npix,size=1000).astype(int)
signal_map[pixels] = 200.0 / Omega_p #*u.mJy/u.sr # units of mJy/sr

#equiv = u.thermodynamic_temperature(90*u.GHz, Planck15.Tcmb0)
#s = signal_map.to(u.uK, equivalencies=equiv)

# Now lets transform to uK thermo units
# I dont do it with the units module because healpy wont let me run arrays that have units in them, so I have to do it manually
# This is taken from the flux_factor function in pixell, https://github.com/simonsobs/pixell/blob/136bdc41d71687aec9726ba041f40312d9f891fc/pixell/utils.py#L1618
freq = 90e9 # We assume that we are working at 90 GHz
x     = h*freq/(k*T_cmb)
dIdT  = 2*x**4 * k**3*T_cmb**2/(h**2*c**2)/(4*np.sinh(x/2)**2)
dJydK = dIdT * 1e26 / 1e3 # This is the factor between spectral radiance and temperature. The 1e26 is because 1 Jy = 1e-26 Watt/meter^2/Hertz , and the 1/1e3 is because we want to work from 
# mJy/sr to uK, so 1e3/1e6 = 1/1e3. You can check this conversion using the astropy units module which has a built in unit convertor, and it seems to be ok.
# I divide because I am going from mJy/sr to uK, not the other way around.
s = signal_map / dJydK
# now s in in units of uK thermo

# We will "observe" with a 30 arcmin beam
fwhm = 30.0
# smoothing convolves the map with a gaussian beam, and it DOES mantain the average of the map
signal_conv_map = hp.smoothing(s,fwhm = np.radians(fwhm/60.0),pol=False)

#print(np.sum(signal_map),np.sum(signal_conv_map))

b_ell = hp.gauss_beam(np.radians(fwhm/60.0),pol=False,lmax=(3*nside-1))

# Sum the point sources map and the noise map
total_map = signal_conv_map + noise_map

norm2 = np.sum( (2*ell_arr + 1 ) * b_ell ) / (4*np.pi) 

# These 2 lines are in Kevins code
#normalization =  4*pi/sum((2*l+1) * bl**2 / Nl) #  ????
#mfl = normalization * bl/Nl

# Matched filter, I use the noise spectra as the filter
# this is the denominator in the matched filter equation
norm  = np.sum( (2*ell_arr + 1 ) * b_ell**2 / Nell) / (4*np.pi) 
#norm  = np.sum( b_ell**2 / Nell) 

# This is the filter 
f_ell =  norm2 * b_ell / Nell / norm

# I need the harmonic transform of the map
alms_tot = hp.map2alm(total_map,pol=False)
# I multiply the alms with the matched filter
alms_tot = hp.almxfl(alms_tot,f_ell)
# transform back to real space
total_MF_map = hp.alm2map(alms_tot,nside)

#The beam area Omega_B, in sr
OmegaB = (np.pi / 4 / np.log(2)) * np.radians(fwhm/60)**2

# total_MF_map is in uK/sr, so we need to transform back to mJy/sr now multiplying for the dJydK factor, and also multiplying by the beam solid angle in sr
total_MF_map = total_MF_map*dJydK*OmegaB

# Now total_MF_map is in mJy
hp.gnomview(total_MF_map,reso=5,title="Match filtered")
pl.show()
