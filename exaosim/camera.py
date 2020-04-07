import numpy as np
import scipy.ndimage as nd
import scipy.fftpack as fftpack
import matplotlib.pyplot as plt
from astropy.io import fits
from .utils import fft_img, rebin, reshape

pi = np.pi


class CameraSystem(object):
    """Simulation of a camera system with lens and camera.
    The camera can take image of extended target such as solar surface.
    """

    def __init__(self, pupil, position=[0, 0], plate_scale=0.01, plate_pix=512,
                 image_enhance=2, wavelength=500E-9):
        """initial the camera system

        Args:
            pupil (Pupil): a Pupil object content the information of the pupil
            position (list, optional): position of the aperture. Defaults to
            [0, 0].
            plate_size (float, optional): phsical size of the focal plane.
            Defaults to 5.12.
            plate_pix (int, optional): pixel size of the focal plane. Defaults
            to 512.
            image_enhance (int, optional): the real image size will be
            plate_pix * image_enhance, then bin to plate_pix. Defaults to 1.
            wavelength ([type], optional): wavelength. Defaults to 500E-9.
        """

        self.plate_size = plate_pix * plate_scale
        self.plate_pix = plate_pix

        self.image_enhance = image_enhance

        self.wavelength = wavelength

        self.pupil = pupil

        self.pupil_size = pupil.size  # size means physical size
        self.pupil_pix = pupil.pixsize  # pixsize means pixel size

        self.position = position

        self.plate_scal = self.plate_size/self.plate_pix
        self.pupil_scal = self.pupil_size/self.pupil_pix

    def run(self, phase, target):
        return self.take_image_wv(phase, target)

    def cut(self, psf, size, center=[0, 0]):
        '''
        cut the image to a certen size, base on the center of the image. Used in Camera Object.
        '''
        sz = psf.shape
        xs = 0
        ys = 0
        if sz[0] % 2 != size[0] % 2:
            xs = -0.5
        if sz[1] % 2 != size[1] % 2:
            ys = -0.5
        psf1 = nd.interpolation.shift(psf, (xs + center[0], ys + center[1]))
        psf = np.zeros(size, dtype=float)

        if size[0] < sz[0]:
            xstart = int((sz[0] - size[0]) / 2 + xs)
            ystart = int((sz[1] - size[1]) / 2 + ys)
            return psf1[xstart:xstart+size[0], ystart:ystart+size[1]]
        else:
            xstart = -int((sz[0] - size[0]) / 2 + xs)
            ystart = -int((sz[1] - size[1]) / 2 + ys)
            psf[xstart:xstart+sz[0], ystart:ystart+sz[1]] = psf1
            return psf

    def take_image_wv(self, phase, target):
        if phase.shape[0] != self.pupil.pixsize:
            raise Exception('Phase Size Error')
        
        pos = np.array(target.pos)
        img = fft_img(self.pupil.pupil, self.pupil_scal, self.pupil_size,
                      self.plate_scal, wavelength=self.wavelength,
                      image_enhance=self.image_enhance, phase=phase)
        fits.writeto('realimg.fits', img, overwrite=True)
        ps = int(self.plate_pix)
        cutimg = self.cut_psf(img, [ps, ps],
                          center=pos[:2] / self.plate_scal) ** 2

        return cutimg/cutimg.sum()*target.magnitude

    def centroid(self, image, initial_gauss=[0,0]):
        cal_size = image.shape[0]
        x = np.arange(cal_size) - (cal_size - 1)/2
        xx, yy = np.meshgrid(x, x)
        x = np.sum(xx * image)/np.sum(image)
        y = np.sum(yy * image)/np.sum(image)
        return (x, y)

    def cut_psf(self, psf, size, center=[0, 0], centroid=False):
        '''
        cut the image to a certen size, base on the center of the image. Used in Camera Object.
        '''
        sz = psf.shape
        xs = 0
        ys = 0

        if sz[0] % 2 != size[0] % 2:
            xs = -0.5
        if sz[1] % 2 != size[1] % 2:
            ys = -0.5

        xc = 0
        yc = 0
        if centroid:
            xc, yc = self.centroid(psf)

        psf1 = nd.interpolation.shift(psf, (xs + center[0] - xc, ys + center[1] - yc))
        psf = np.zeros(size, dtype=float)

        if size[0] < sz[0]:
            xstart = int((sz[0] - size[0]) / 2 + xs)
            ystart = int((sz[1] - size[1]) / 2 + ys)
            return psf1[xstart:xstart+size[0], ystart:ystart+size[1]]
        else:
            xstart = -int((sz[0] - size[0]) / 2 + xs)
            ystart = -int((sz[1] - size[1]) / 2 + ys)
            psf[xstart:xstart+sz[0], ystart:ystart+sz[1]] = psf1
            return psf


class CameraSystem2(object):
    """Simulation of a camera system with lens and camera.
    The camera can take image of extended target such as solar surface.
    """

    def __init__(self, pupil, position=[0, 0], plate_scale=0.01, plate_pix=512,
                wavelength=500E-9, bin=1):
        """initial the camera system

        Args:
            pupil (Pupil): a Pupil object content the information of the pupil
            position (list, optional): position of the aperture. Defaults to
            [0, 0].
            plate_size (float, optional): phsical size of the focal plane.
            Defaults to 5.12.
            plate_pix (int, optional): pixel size of the focal plane. Defaults
            to 512.
            image_enhance (int, optional): the real image size will be
            plate_pix * image_enhance, then bin to plate_pix. Defaults to 1.
            wavelength ([type], optional): wavelength. Defaults to 500E-9.
        """

        self.plate_size = plate_pix * plate_scale
        self.plate_pix = plate_pix

        self.wavelength = wavelength

        self.pupil = pupil

        self.pupil_size = pupil.size  # size means physical size
        self.pupil_pix = pupil.pixsize  # pixsize means pixel size
        self.auto_enhance = 2

        self.position = position
        self.input_plate_scal = self.plate_size/self.plate_pix
        self.input_plate_pix = self.plate_pix
        self.plate_scal = self.plate_size/self.plate_pix
        self.pupil_scal = self.pupil_size/self.pupil_pix

        self.bin = bin

    def run(self, phase, target):
        return self.take_image_wv(phase, target)

    def psf_fft(self, phase):
        pupil_image = self.pupil.pupil
        wavelength = self.wavelength
        wl = wavelength
        sz = pupil_image.shape[0]

        wl_rate = 500E-9/wl
        pupil_scal = self.pupil_scal
        plate_scal = self.input_plate_scal

        n_big = wl*180.0*3600.0/pi / pupil_scal / plate_scal * self.bin
        print(n_big)
        N_big = int(n_big)

        if N_big < self.auto_enhance * sz:
            N_big = self.auto_enhance * sz
            self.enhanced = True
            
        N_big += 1 - N_big%2

        realscale = wl*180.*3600./pi/N_big/pupil_scal
        bigimg = np.zeros((N_big, N_big), dtype=complex)
        complex_phase = np.cos(phase*wl_rate) + 1j*np.sin(phase*wl_rate)
        bigimg[0:sz, 0:sz] = pupil_image * complex_phase
        fftimg = fftpack.fft2(bigimg)
        fftreal = fftpack.fftshift(abs(fftimg))
        return fftreal, realscale

    def image_trans(self, image, realscale, shifts):
        n_out = self.bin * self.plate_pix
        output_scale = self.plate_scal / self.bin
        rate = realscale / output_scale
        # rate > 1, zoom out
        # rate < 1, zoom in
        n = image.shape[0]
        shiftx = shifts[0]
        shifty = shifts[1]

        def coor_change(coor_in):
            out0 = (coor_in[0] - (n_out - 1) / 2 - shifty) * rate + (n - 1)/2
            out1 = (coor_in[1] - (n_out - 1) / 2 - shiftx) * rate + (n - 1)/2
            return (out0, out1)

        res = nd.interpolation.geometric_transform(image, coor_change)[:n_out, :n_out]
        return rebin(res[:n_out, :n_out], self.bin)


    def take_image_wv(self, phase, target):
        if phase.shape[0] != self.pupil.pixsize:
            raise Exception('Phase Size Error')
        
        pos = np.array(target.pos)
        img, real_scale = self.psf_fft(phase)
        img = self.image_trans(img, real_scale, pos)
        # fits.writeto('realimg.fits', img, overwrite=True)

        return img/img.sum()*target.magnitude

    def centroid(self, image):
        cal_size = image.shape[0]
        x = np.arange(cal_size) - (cal_size - 1)/2
        xx, yy = np.meshgrid(x, x)
        x = np.sum(xx * image)/np.sum(image)
        y = np.sum(yy * image)/np.sum(image)
        return (x, y)

    def cenfit(self, image):
        my, mx = np.unravel_index(np.argmax(image), image.shape)
        sub = np.roll(image, (-my+1, -mx+1), axis=(0,1))
        d = sub.mean(axis=1)
        sy = (d[2] - d[0])/(2 * d[1] - d[0] - d[2])/2# - (img_sz - 1)/2
        d = sub.mean(axis=0)
        sx = (d[2] - d[0])/(2 * d[1] - d[0] - d[2])/2
        img_sp = image.shape
        return mx + sx - (img_sp[1] - 1)/2, my + sy - (img_sp[0] - 1)/2




class Pupil(object):
    def __init__(self, d, pixsize, scale, shift=[0, 0]):
        self.pixsize = int(pixsize)
        self.size = self.pixsize * scale
        self.scale = scale
        self.shift = shift
        self.img = None
        self.area = None
        self.d = d
        self.type = 'defalt'

    def make_pupil(self, enhance=1):
        size = self.size
        pixsize = int(self.pixsize * enhance)
        scale = size / pixsize
        x = np.arange(pixsize) - (pixsize - 1)/2
        xx, yy = np.meshgrid(x, x)
        pupil = (abs(xx) < (self.d / scale / 2)).astype(int)
        pupil = pupil * pupil.T
        self.pupil = pupil
        self.pupil = reshape(pupil, self.scale/scale)

    def load_from(self, file):
        img, header = fits.getdata(file, header=True)
        self.pixsize = img.shape[0]
        self.pupil = img
        self.size = header['size']
        self.d = header['diameter']
        self.type = header['type']

    def save_to(self, file):
        head = fits.Header()
        head['diameter'] = self.d
        head['size'] = self.size
        head['type'] = self.type
        fits.writeto(file, self.pupil, header=head, overwrite=True)

    def reduce_rate(self, rate):
        if rate > 1:
            img = self.pupil
            sz = np.array(img.shape)
            fimg = np.fft.fftshift(np.fft.fft2(img))
            new_sz = (sz/rate).astype(int)
            ori = ((sz-new_sz)/2).astype(int)

            sub_fimg = fimg[ori[0]:ori[0]+new_sz[0], ori[1]:ori[1]+new_sz[1]]
            sub_img = np.fft.ifft2(sub_fimg)
            rate = new_sz[0]/self.pixsize
            self.pupil = abs(sub_img) / rate**2

        elif rate < 1:
            raise Exception('Cant imcrease rate!')

    def show(self, ax=plt):
        ax.imshow(self.pupil.T, origin='lowerdef ')


class SqrPupil(Pupil):
    def __init__(self, d, pixsize, scale, shift=[0, 0]):
        super().__init__(d, pixsize, scale, shift=shift)

    def make_pupil(self, enhance=1):
        size = self.size
        pixsize = int(self.pixsize * enhance)
        scale = size / pixsize
        x = np.arange(pixsize) - (pixsize - 1)/2
        xx, yy = np.meshgrid(x, x)
        pupil = (abs(xx) < (self.d / scale / 2 + 0.5)**2).astype(int)
        self.pupil = pupil * pupil.T
        if enhance > 1:
            self.reduce_rate(enhance)

class CirclePupil(Pupil):
    def __init__(self, d, pixsize, scale, shift=[0, 0]):
        super().__init__(d, pixsize, scale, shift=shift)

    def make_pupil(self, enhance=1):
        size = self.size
        pixsize = int(self.pixsize * enhance)
        scale = size / pixsize
        x = np.arange(pixsize) - (pixsize - 1)/2
        xx, yy = np.meshgrid(x, x)
        rr = xx**2 + yy**2
        pupil = (abs(rr) < (self.d / scale / 2 + 0.5)**2).astype(int)
        self.pupil = pupil
        self.reduce_rate(enhance)
