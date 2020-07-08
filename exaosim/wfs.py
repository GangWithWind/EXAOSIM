import copy
import numpy as np
import scipy.fftpack as fftpack
import matplotlib.pyplot as plt
from .camera import CameraSystem2, SqrPupil
from .port import read_c_array, write_c_array
from astropy.io import fits
import ctypes

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
        self.gpu_turbo = True

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

    def make_sub_aperture(self, edge_effect=False):
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
# for debug
        debug_image = []

        for ap in self.apertures:
            pos = (ph_cen + ap.position / ap.pupil_scal - sz / 2).astype(int)
            sub_phase = phase[pos[0]: pos[0]+sz, pos[1]: pos[1]+sz]
            complex_phase = np.cos(sub_phase*wl_rate) + 1j*np.sin(sub_phase*wl_rate)
            bigimg[:sz, :sz] = ap.pupil.pupil * complex_phase
            fftimg = fftpack.fft2(bigimg)
            # debug_image.append(abs(fftimg)**2)
            fftreal = fftpack.fftshift(abs(fftimg))
            images.append(fftreal[ps:ps+ppx, ps:ps+ppx]**2)

        # fits.writeto("debug_middle.fits", np.array(debug_image), overwrite=True) ##

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


    def cuda_run(self, phase):
        res = np.zeros(self.wfs_img_sz * self.wfs_img_sz, dtype=np.float32)
        phase = phase.astype(np.float32).flatten()
        self.cu_lib.cuwfs_run(res.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                              phase.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 0)
                              
        return res.reshape((self.wfs_img_sz, self.wfs_img_sz))


    def cuda_destroy(self):
        self.cu_lib.cuwfs_destroy()

    def cuda_plan(self, phase_size):

        self.cu_lib = ctypes.cdll.LoadLibrary("./lib/libcuwfs2.so")

        ph_sz = (self.n_grid + 1) * self.aperture_pix
        ph_cen = (ph_sz - 1)/2
        sz = self.aperture_pix
        subs = []
        pupils = []

        for ap in self.apertures:
            pos = (ph_cen + ap.position / ap.pupil_scal - sz / 2).astype(int)
            subs.append(pos)
            pupils.append(ap.pupil.pupil)

        subs = np.array(subs, dtype=np.int32).flatten()
        pupils = np.array(pupils, dtype=np.int32).flatten()

        edge = (self.plate_pix - self.plate_interval)//2
        bigsz = self.plate_interval * self.n_grid + edge * 2
        big_index = {}
        self.wfs_img_sz = bigsz

        n_bin = 1
        n_sub = len(self.apertures)
        n_fft = self.fast_Nbig * n_bin
        
        n_sfft = self.fast_Nbig
        fft_index = np.arange(n_fft * n_fft * n_sub, dtype=int)
        fft_index = fft_index.reshape((n_sub, n_fft, n_fft))

        for i in range(n_sub):
            fft_index[i, :, :] = np.fft.fftshift(fft_index[i, :, :])
        
        fft_index = fft_index.reshape((n_sub, n_sfft, n_bin, n_sfft, n_bin))
        fft_index = np.swapaxes(fft_index, 2, 3)
        fft_index = fft_index.reshape((n_sub, n_sfft, n_sfft, n_bin * n_bin))

        n = self.plate_pix // self.plate_interval + 1

        ppx = self.plate_pix
        ps = (n_sfft - ppx)//2

        for i, ap in enumerate(self.apertures):
            pos = (np.array(ap.position) / self.aperture_size) * self.plate_interval + (bigsz - self.plate_interval)/2 - edge
            pos = pos.astype(int)

            for y in range(self.plate_pix):
                for x in range(self.plate_pix):
                    b = big_index.get((y + pos[1], x + pos[0]), [])
                    for z in fft_index[i, y + ps, x + ps, :]:
                        b.append(z)
                    big_index[(y + pos[1], x + pos[0])] = b

        ni_max = 0
        for item in big_index.items():
            ni = len(item[1])
            if ni > ni_max:
                ni_max = ni

        a_big_index = np.zeros((bigsz, bigsz, ni_max)) - 1

        for yi, xi in big_index.keys():
            d = big_index[(yi, xi)]
            ni = len(d)
            a_big_index[yi, xi, 0: ni] = np.array(d)
       
        print('bs', a_big_index.shape)

        patch = a_big_index.astype(np.int32).flatten()

        # fits.writeto("big_index.fits", a_big_index[:,:,0].astype(int), overwrite=True)
        # debug_big = np.zeros((bigsz, bigsz))
        # fft_out = read_c_array('middle.bin')
        # fits.writeto("fftout.fits", fft_out, overwrite=True)

        # fft_out = fft_out.flatten()
        # for i in range(bigsz):
        #     for j in range(bigsz):
        #         ind = int(a_big_index[i, j, 0])
        #         if ind >= 0:
        #             debug_big[i, j] = fft_out[ind]

        # fits.writeto('python_patch.fits', debug_big, overwrite=True)

        self.cu_lib.cuwfs_plan(int(n_fft),
                               int(self.aperture_pix),
                               int(n_sub),
                               int(ph_sz),
                               int(bigsz),
                               int(ni_max),
                               subs.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                               pupils.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                               patch.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
                               )


    def cuda_plan_old(self):

        ph_sz = (self.n_grid + 1) * self.aperture_pix
        ph_cen = (ph_sz - 1)/2
        sz = self.aperture_pix
        subs = []
        pupils = []

        for ap in self.apertures:
            pos = (ph_cen + ap.position / ap.pupil_scal - sz / 2).astype(int)
            subs.append(pos)
            pupils.append(ap.pupil.pupil)

        subs = np.array(subs)
        pupils = np.array(pupils)
        
        write_c_array('../testdata/wfs_subs.bin', subs.astype(float))
        write_c_array('../testdata/wfs_pupils.bin', pupils)

        edge = (self.plate_pix - self.plate_interval)//2
        bigsz = self.plate_interval * self.n_grid + edge * 2
        big_index = {}

        n_bin = 1
        n_sub = len(self.apertures)
        n_fft = self.fast_Nbig * n_bin
        write_c_array('../testdata/wfs_nbig.bin', np.array(n_fft).astype(float))
        
        n_sfft = self.fast_Nbig
        fft_index = np.arange(n_fft * n_fft * n_sub, dtype=int)
        fft_index = fft_index.reshape((n_sub, n_fft, n_fft))

        for i in range(n_sub):
            fft_index[i, :, :] = np.fft.fftshift(fft_index[i, :, :])
        
        fft_index = fft_index.reshape((n_sub, n_sfft, n_bin, n_sfft, n_bin))
        fft_index = np.swapaxes(fft_index, 2, 3)
        fft_index = fft_index.reshape((n_sub, n_sfft, n_sfft, n_bin * n_bin))

        n = self.plate_pix // self.plate_interval + 1

        ppx = self.plate_pix
        ps = (n_sfft - ppx)//2

        for i, ap in enumerate(self.apertures):
            pos = (np.array(ap.position) / self.aperture_size) * self.plate_interval + (bigsz - self.plate_interval)/2 - edge
            pos = pos.astype(int)

            for y in range(self.plate_pix):
                for x in range(self.plate_pix):
                    b = big_index.get((y + pos[0], x + pos[1]), [])
                    for z in fft_index[i, y + ps, x + ps, :]:
                        b.append(z)
                    big_index[(y + pos[0], x + pos[1])] = b

        ni_max = 0
        for item in big_index.items():
            ni = len(item[1])
            if ni > ni_max:
                ni_max = ni
        a_big_index = np.zeros((bigsz, bigsz, ni_max)) - 1

        for yi, xi in big_index.keys():
            d = big_index[(yi, xi)]
            ni = len(d)
            a_big_index[yi, xi, 0: ni] = np.array(d)

        write_c_array('../testdata/wfs_patch_index.bin', a_big_index.astype(float))
