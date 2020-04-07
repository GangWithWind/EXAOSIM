import copy
import numpy as np
import scipy.fftpack as fftpack
import matplotlib.pyplot as plt
from .camera import CameraSystem2, SqrPupil


class WFS(object):
    def __init__(self):
        self.n_grid = 10
        self.used = []
        self.aperture_pix = 30
        self.aperture_size = 2 / self.n_grid
        self.wavelength = 500E-9
        self.plate_pix = 20
        self.plate_scale = 0.1
        self.plate_interval = 10
        self.whole_pupil_d = 2
        self.apertures = []

    def sub_aperture_config(self, plot=False):
        x = (np.arange(self.n_grid) - (self.n_grid - 1)/2) * self.aperture_size
        xx, yy = np.meshgrid(x, x)
        rr = np.sqrt(xx * xx + yy * yy).flatten()
        self.used = np.where(rr < (self.whole_pupil_d / 2))[0]

        if plot:
            xx = xx.flatten()
            yy = yy.flatten()
            for i in self.used:
                xc = xx[i]
                yc = yy[i]
                d = self.aperture_size/2
                plt.plot([xc - d, xc - d, xc + d, xc + d, xc - d], 
                [yc - d, yc + d, yc + d, yc - d, yc - d], 'k')
                count = 1000
                xarr=[]
                yarr=[]
                r = self.whole_pupil_d/2
                for i in range(count):
                    j = float(i)/count * 2 * np.pi
                    xarr.append(r*np.cos(j))
                    yarr.append(r*np.sin(j))
                plt.plot(xarr, yarr, 'k')

            plt.show()

    def make_sub_aperture(self, edge_effect = False):
        x = (np.arange(self.n_grid) - (self.n_grid - 1)/2) * self.aperture_size
        xx, yy = np.meshgrid(x, x)
        xx = xx.flatten()
        yy = yy.flatten()

        scale = self.aperture_size / self.aperture_pix
        pupil = SqrPupil(self.aperture_size, self.aperture_pix , scale)
        pupil.make_pupil()
        ccd0 = CameraSystem2(pupil, plate_scale=self.plate_scale, plate_pix=self.plate_pix,
                            wavelength=self.wavelength)

        psize = ccd0.pupil_pix
        x = (np.arange(psize) - (psize - 1)/2) * ccd0.pupil_scal
        pyy, pxx = np.meshgrid(x, x)

        for i in self.used:
            ccd = copy.deepcopy(ccd0)
            ccd.position = np.array((xx[i], yy[i]))
            if edge_effect:
                prr = np.sqrt((pxx + xx[i])**2 + (pyy + yy[i])**2)
                ccd.pupil.pupil *= (prr < (self.whole_pupil_d / 2)).astype(int)
            self.apertures.append(ccd)

    def shift_from_slope(self, phase):
        pupil_scal = self.apertures[0].pupil_scal
        shift_factor = self.wavelength / (np.pi / 180 / 3600) / 2 / np.pi / pupil_scal

        # print("inner slope", np.mean(phase[:, -1] - phase[:, 0]) / (phase.shape[1]-1))
        ys = np.mean(phase[:, -1] - phase[:, 0]) / (phase.shape[1]-1) * shift_factor
        xs = np.mean(phase[-1, :] - phase[0, :]) / (phase.shape[0]-1) * shift_factor
        return (xs, ys)


    def take_image_fast(self, phase, target):
        ap = self.apertures[0]
        ph_sz = phase.shape[0]
        ph_cen = (ph_sz - 1)/2

        N_big = self.fast_Nbig
        if N_big % 2 != 1:
            print('N big must be odd')

        if self.plate_pix % 2 != 1:
            print('plate_pix must be odd')

        self.plate_scale = self.wavelength*180.*3600./np.pi/N_big/ap.pupil_scal
        bigimg = np.zeros((N_big, N_big), dtype=complex)
        wl_rate = self.wavelength/500E-9
        sz = self.aperture_pix
        ppx = self.plate_pix
        ps = (N_big - ppx)//2
        images = []

        for ap in self.apertures:
            pos = (ph_cen + ap.position / ap.pupil_scal - sz / 2).astype(int)
            sub_phase = phase[pos[0]: pos[0]+sz, pos[1]: pos[1]+sz]
            complex_phase = np.cos(sub_phase*wl_rate) + 1j*np.sin(sub_phase*wl_rate)
            bigimg[:sz, :sz] = ap.pupil.pupil * complex_phase
            fftimg = fftpack.fft2(bigimg)
            fftreal = fftpack.fftshift(abs(fftimg))
            images.append(fftreal[ps:ps+ppx, ps:ps+ppx])

        return self.patch_images(images)

    def take_image(self, phase, target, shifts=False, slope_shifts=False):
        ph_sz = phase.shape[0]
        ph_cen = (ph_sz - 1)/2

        wfssz = self.n_grid * self.aperture_pix
        images = []

        sss = []
        for ap in self.apertures:
            subp_sz = ap.pupil_pix
            pos = (ph_cen + ap.position / ap.pupil_scal - subp_sz / 2).astype(int)
            sub_phase = phase[pos[0]: pos[0]+subp_sz, pos[1]: pos[1]+subp_sz]
            images.append(ap.run(sub_phase, target))
            # print(ap.centroid(images[-1]))
            if slope_shifts:
                sss.append(self.shift_from_slope(sub_phase))

        if shifts:
            shifts_value = []
            ccd = self.apertures[0]

            for img in images:
                shifts_value.append(ccd.cenfit(img))
            if slope_shifts:
                return self.patch_images(images), shifts_value, sss

            return self.patch_images(images), shifts_value

        return self.patch_images(images)

    def patch_images(self, images):
        edge = (self.plate_pix - self.plate_interval)//2
        bigsz = self.plate_interval * self.n_grid + edge * 2

        bigimg = np.zeros((bigsz, bigsz))

        for i, ap in enumerate(self.apertures):
            pos = (np.array(ap.position) / self.aperture_size) * self.plate_interval + (bigsz - self.plate_interval)/2 - edge
            pos = pos.astype(int)
            bigimg[pos[0]: pos[0]+self.plate_pix, pos[1]: pos[1]+self.plate_pix] += images[i]

        return bigimg
