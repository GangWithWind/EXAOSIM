import numpy as np
import scipy.ndimage as nd

pi = np.pi


def primefactors(n):
    """Generate all prime factors of n. used in the camera modula to find the 
    fastest fft size.

    Args:
        n (int): input n

    Returns:
        int: count of prime factors of input target n
    """
    f = 2
    nres = 0
    while f * f <= n:
        while not n % f:
            nres += 1
            n //= f
        f += 1
    if n > 1:
        nres += 1
    return nres


def reshape(psf, rate, anti_aliasing=0):
    '''
    zoom in and zoom out base on the center of the image. using order 3
    spline interpolation. This function is used in the Camera Object,
    to adjust the pixel rate. The size of image will not change.
    
    input:  
        (np.array) psf: input image
        (float) rate: zoom rate, >1, zoom out, <1, zoom in.
    keywords:
        (int) anti_aliasing: method for anti_aliasing (not finished)
    output:  
        (np.array): zoomed image. 
    '''
    if anti_aliasing == 1:
        fre = np.fft.fftshift(np.fft.fft2(psf))
        dx = int(sz[0] * (1 - rate) /2)
        fre[:dx,0] = 0
        fre[-dx:,0] = 0
        fre[:0,dx] = 0
        fre[0:-dx:] = 0
        psf = abs(np.fft.ifft2(fre))
     
    if anti_aliasing == 2:
        xbin = 1
        if rate >= 0.5:
            xbin = 2
        psf = rebin(psf,xbin)
        rate /= xbin

    sz = psf.shape
        
    def coor_change(coor_in):
        out0 = coor_in[0] * rate + ((sz[0] - 1) / 2) * (1 - rate)
        out1 = coor_in[1] * rate + ((sz[1] - 1) / 2) * (1 - rate)
        return (out0, out1)
    res=nd.interpolation.geometric_transform(psf,coor_change)
    return res


def fft_img(pupil_image, pupil_scal, pupil_size, plate_scal,
            wavelength=500e-9, image_enhance=2, phase=np.array(0), reshpae=True):

    wl = wavelength
    sz = pupil_image.shape[0]
    pupil_size = np.array(pupil_size)

    wr = 500E-9/wl

    n_big = wl*180.0*3600.0/pi / pupil_scal / plate_scal

    print('plate_scal', plate_scal)
    print('pupil_scal', pupil_scal)
    print('nbig', n_big)
    
    N_big = np.maximum(int(n_big), image_enhance * sz)

    # change the N_big size for following reason:
    # 1.N_big is need to be even, which is easy to be cut
    # to the pixsize of CCD which is more likely to be even
    # And also it is recommand to set the pixsize of CCD to be even
    N_big = N_big + N_big % 2
    # 2.searching in the interval [N_big, N_big + 18]
    # and find the value with most prime factors, which will increase the 
    # speed of FFT

    N_big_list = N_big + np.arange(10)*2
    N_big = max(N_big_list, key=primefactors)

    realscale = wl*180.*3600./pi/N_big/pupil_scal
    rate = plate_scal/realscale

    bigimg = np.zeros((N_big, N_big), dtype=complex)

    complex_phase = np.cos(phase*wr) + 1j*np.sin(phase*wr)
    bigimg[0:sz, 0:sz] = pupil_image * complex_phase

    # TODO update using pyFFTW and speed tests
    fftimg = np.fft.fft2(bigimg)
    fftreal = np.fft.fftshift(abs(fftimg))
    res = reshape(fftreal[1:, 1:], rate)
    return res


def rebin(a, factor):
    '''binning of an image, can only do n*n binning
    input: 
        (np.array) a : image
        (int) factor : factor of binning
    output:
        (np.array) : image after binning
    '''
    shape = a.shape
    binshape = (np.asarray(shape)/factor).astype(int)
    subshape = binshape * factor
    sub = a[0:subshape[0],0:subshape[1]]
    return sub.reshape(binshape[0],factor,binshape[0],factor).mean(1).mean(2)