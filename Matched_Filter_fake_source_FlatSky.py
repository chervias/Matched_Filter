from pixell import enmap,enplot,utils
import matplotlib.pyplot as pl
import numpy as np
from astropy.convolution import convolve, Gaussian2DKernel
from astropy import units as u
from astropy.cosmology import Planck15

# 3deg by 3deg map with 0.5 arcmin pixels, so 360 by 360 pixels
m = np.zeros((360,360))*u.mJy/u.sr
# Inject a fake source with 200 mJy, because this is a delta model, the flux density per solid angle will be 200 mJy / (area of each pixel)
m[180,180] = 200 / np.radians(0.5/60.0)**2 *u.mJy/u.sr

# this is to transform Jy/(solid angle) to thermo kelvin
equiv = u.thermodynamic_temperature(90*u.GHz, Planck15.Tcmb0)
m2 = m.to(u.uK, equivalencies=equiv)

# I will convolve the image with a gaussian beam of 2 arcmin FWHM. The sigma of that gaussian is 2/2.355. Because astropy works with kernels in pixel space, we divide by 0.5 arcmin per pixel
sigma = 2.0/2.355 / 0.5
print('The sigma of the gauss kernel is ',sigma,' pixels')

kernel = Gaussian2DKernel(sigma)
m_conv = convolve(m2, kernel)
# make sure the convolution conserves the mean of the map
print('Is the mean conserved by the convolution ? ',np.sum(m2),np.sum(m_conv))

pl.imshow(m_conv[160:200,160:200],vmin=0.0,vmax=3000)
pl.colorbar()
#pl.show()

# Now I load an enmap map that is 3deg by 3deg with 0.5 arcmin per pixel, and I replace its values with the values on m_conv
# I have to do the matched filter with an enmap map, since the fft is just a line of code because it already has the WCS info embedded
filename = 'out_QSO_B2332-017/sky_map0020.fits'
m = enmap.read_map(filename)[0]
for i in range(360):
	for j in range(360):
		m[i,j] = m_conv[i,j]

ell = m.modlmap()
ell2 = m.lmap()
dk1 = ell2[0,300,300] - ell2[0,299,300]
dk2 = ell2[1,300,300] - ell2[1,300,299]

print('dk',dk1,dk2)

beam_2d_m = np.exp(-0.5*ell**2 * (2.0*utils.fwhm*utils.arcmin)**2)
factor = np.sum(beam_2d_m)*dk1*dk2

# this is the matched filter
dk_squared = dk1*dk2
norm = np.sum(beam_2d_m**2 / (1+(ell/2000)**-3))*dk_squared
f = factor * beam_2d_m / (1+(ell/2000)**-3) / norm

m_filtered = enmap.ifft(f*enmap.fft(m)).real

b_area = np.sum(kernel.array / np.max(kernel.array) )*np.radians(0.5/60.0)**2 * 1e9
print('The beam area is ',b_area,' nsr')

fac = utils.flux_factor(b_area*1e-9,90*1e9)/1e3 # this factor is K to Jy, the division by /1e3 is to get it to uK to mJy

m_mJy = fac*m_filtered

#pl.imshow(m_mJy)
#pl.colorbar()

print('The measured flux of the source is ',m_mJy[180,180],'mJy')
