import env
import numpy as np
from scipy.special import jv
from exaosim.camera import *
from exaosim.target import *
import matplotlib.pyplot as plt
from astropy.io import fits
pi = np.pi
r2a = 180 * 3600 / pi

def psf(theta, wavelen=500E-9, d=1):
    u = pi * theta / (wavelen / d / pi *180 *3600)
    print('l2d', (wavelen / d / pi *180 *3600) )
    print('npi', 100 /(wavelen / d / pi *180 *3600))

    #return (np.sin(u) / u)**2
    return (jv(1, u)*2/u)**2

def test_cut():
    img = fits.getdata('tests/a.fits')
    xs = 0
    ys = 0
    img  = nd.interpolation.shift(img, (ys, xs))
    pupil = Pupil(0.1, 55, 0.0022)
    pupil.make_pupil()
    ccd = CameraSystem(pupil)
    img = ccd.cut_psf(img, (50,50), centroid=True)
    fits.writeto('as.fits', img, overwrite=True)
    print(ccd.centroid(img, img.shape[0]))
    #TODO test the center of the image by utils.fft


def test_make_camera():
    rate = 4
    pupil = Pupil(0.1, 55 * rate, 0.0022 / rate)
    print(500e-9/0.1 / np.pi *180 *3600)
    pupil.make_pupil()
    # pupil.reduce_rate(rate)
    pupil.save_to('p.fits')
    phase = np.zeros((pupil.pixsize, pupil.pixsize))
    ccd = CameraSystem(pupil, plate_scale=0.1)
    target = Target()
    img = ccd.run(phase, target)
    fits.writeto('a.fits', img, overwrite=True)
    sz = img.shape
    x = np.arange(sz[0]/2) * ccd.plate_scal
    y = img[sz[0]//2, sz[0]//2:]
    theta = np.linspace(0, 10, 1000)

    l2d = ccd.wavelength/pupil.d * r2a

    psfi = psf(theta, d=pupil.d)
    plt.semilogy(theta/l2d, psfi, label='thrical')
    plt.semilogy(x/l2d, y/y[0])
    plt.legend()
    plt.xlim([0,10])
    plt.show()

def test_circle_pupil_camera():
    rate = 1
    pupil = CirclePupil(0.1, 55*rate, 0.0022/rate)
    pupil.make_pupil(enhance=5)
    pupil.save_to('p.fits')
    phase = np.zeros((pupil.pixsize, pupil.pixsize))
    ccd = CameraSystem(pupil, plate_scale=0.1)
    target = Target()
    img = ccd.run(phase, target)
    fits.writeto('a.fits', img, overwrite=True)
    sz = img.shape
    x = np.arange(sz[0]/2) * ccd.plate_scal
    y = img[sz[0]//2, sz[0]//2:]
    theta = np.linspace(0, 40, 1000)

    l2d = ccd.wavelength/pupil.d * r2a

    psfi = psf(theta, d=pupil.d)
    plt.semilogy(theta/l2d, psfi, label='thrical')
    plt.semilogy(x/l2d, y/y[0])
    plt.legend()
    plt.xlim([0,20])
    plt.show()   

def test_Pupil():
    pupil = SqrPupil(1.0, 1.1, 110)
    pupil.make_pupil()
    pupil.save_to('sqr.fits')
    pupil.load_from('sqr.fits')
    # pupil.reduce_rate(2)
    pupil.show(ax = plt)
    plt.show()

if __name__ == "__main__":
    test_cut()