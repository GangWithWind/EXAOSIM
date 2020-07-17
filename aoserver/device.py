import ctypes
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy import interpolate
from .zernike import zernike, noll_indices


class Aberration(object):
    def __init__(self):
        self.big_phase = []
        self.bigsz = [10, 10]
        self.phase_sz = [4, 4]
        self.now = [0, 0]
        self.step = 2
        self.x_dir = 1
        self.y_dir = 1
        self.time0 = 0
        self.speed = 100

    def initial(self):
        xx, yy = np.meshgrid(np.arange(self.bigsz[0]), np.arange(self.bigsz[1]))
        self.big_phase = xx

    def get_data(self):
        return self.get_data_by_step(self.step)

    def get_data_by_step(self, step):
        if (self.now[0] + self.x_dir * step >
            self.bigsz[0] - self.phase_sz[0] or self.now[0] +
            self.x_dir * step < 0):
            self.x_dir *= -1
            if (self.now[1] + self.y_dir * step >
                self.bigsz[1] - self.phase_sz[1] or self.now[1] +
                self.y_dir * step < 0):
                self.y_dir *= -1
            self.now[1] += self.y_dir * step
        else:
            self.now[0] += step * self.x_dir
        return self.big_phase[self.now[0]:self.now[0] + self.phase_sz[0], self.now[1]:self.now[1] + self.phase_sz[1]]

    def get_data_by_time(self):
        if not self.time0:
            self.time0 = time.perf_counter()
        time_now = time.perf_counter()
        dt = time_now - self.time0
        step = np.minimum(dt * self.speed, 100)
        print(f'Frequence: {1/dt}')
        self.time0 = time_now
        self.phase = self.get_data_by_step(int(step))
        return self.phase


class ZernikeAbrr(Aberration):
    def __init__(self):
        super().__init__()
        self.order = 10

    def initial(self):
        size = (np.array(self.bigsz) * np.sqrt(2) * 1.1).astype(int)
        size = np.max(size)
        ps = np.zeros((size, size))

        for j in range(1, self.order):
            n, m = noll_indices(j)
            ps += np.random.rand() * zernike(n, m, npix = size)
    
        xs = ((size - np.array(self.bigsz))/2).astype(int)
        self.big_phase = ps[xs[0]: xs[0] + self.bigsz[0], xs[1]: xs[1] + self.bigsz[1]]


class TurbAbrr(Aberration):
    def __init__(self):
        super().__init__()
        self.order = 10
        self.r0 = 0.5
        self.L0 = None
        self.sizem = 14

    def get_filter(self):
        freq = self.dist(self.pixsize)/self.sizem
        freq[0, 0] = 1.0
        factors = np.sqrt(0.00058)*self.r0**(-5.0/6.0)
        factors *= np.sqrt(2)*2*np.pi/self.sizem

        if not self.L0:
            self.filter = factors * freq**(-11.0/6.0)
        else:
            self.filter = factors * (freq ** 2 + self.L0**(-2))**(-11.0/12.0)

        self.filter[0, 0] = 0

    def dist(self, pixsize):
        nx = np.arange(pixsize)-pixsize/2
        gxx, gyy = np.meshgrid(nx, nx)
        freq = gxx**2 + gyy**2
        freq = np.sqrt(freq)
        return np.fft.ifftshift(freq)

    def new_phs(self):
        phase = np.random.randn(self.pixsize, self.pixsize)*np.pi
        x_phase = np.cos(phase) + 1j*np.sin(phase)
        pscreen = np.fft.ifft2(x_phase*self.filter)
        ps = np.real(pscreen)*self.pixsize**2
        return ps

    def initial(self):
        self.pixsize = self.bigsz[0]
        self.get_filter()
        self.big_phase = self.new_phs()/2
        self.phase = np.zeros(self.phase_sz)


class DM(object):
    def __init__(self):
        super().__init__()
        self.n_grid = 10
        self.act_pix = 20
        self.used = []
        self.all_act = []
        self.error = []
        self.phase_shift_error = 0
        self.vol_coe_mean = 1
        self.vol_coe_error = 0
        self.file_i = 0
        
    def initial(self):
        acts = np.zeros((self.n_grid + 2, self.n_grid + 2), dtype=float) + 1
        acts[:, 0] = 0
        acts[:, -1] = 0
        acts[0, :] = 0
        acts[-1, :] = 0
        self.n_act = self.n_grid**2
        self.all_act = acts
        self.used = np.where(acts > 0.5)
        self.phase_shift = np.random.randn(self.n_act) * self.phase_shift_error
        self.vol_coe = np.random.randn(self.n_act) * self.vol_coe_error + self.vol_coe_mean
        sz_in = self.all_act.shape[0]
        self.x_in = np.arange(sz_in) - (sz_in - 1)/2
        npix = self.act_pix * self.n_grid
        self.npix = self.act_pix * self.n_grid
        self.x_out = (np.arange(npix) - (npix - 1)/2)/self.act_pix
        self.phase = np.zeros(npix * npix)
        self.vol = np.zeros(npix)
        self.fmt = '%di'%(npix)

    def get_data(self, vol):
        self.all_act[self.used[0], self.used[1]] = vol * self.vol_coe + self.phase_shift
        f = interpolate.interp2d(self.x_in, self.x_in, self.all_act, kind='cubic')
        phase = f(self.x_out, self.x_out).flatten()
        self.phase = phase


class TipTiltMirror(object):
    def __init__(self, pix):
        self.pix = pix
        self.n_act = 2
        self.size = 2
        self.phase = np.zeros(pix * pix)

    def initial(self):
        x = (np.arange(self.pix) - (self.pix - 1)/2)/(self.pix - 1) * 2
        yy, xx = np.meshgrid(x, x)
        self.yy = yy.astype(np.float32).flatten() * 10
        self.xx = xx.astype(np.float32).flatten() * 10

    def get_data(self, vol):
        self.phase = self.xx * vol[0] + self.yy * vol[1] 



class SubAperture(object):
    def __init__(self):
        self.pos = []
        self.pupil = []


class WFS(object):
    def __init__(self):
        super().__init__()
        self.n_grid = 10
        self.used = []
        self.aperture_pix = 30
        self.aperture_size = 2 / self.n_grid
        self.wavelength = 500E-9
        self.plate_pix = 21
        self.plate_scale = 0.1
        self.plate_interval = 11
        self.whole_pupil_d = 2
        self.apertures = []
        self.gpu_turbo = True
        self.phase_pix = self.aperture_pix * (self.n_grid + 1)
        self.rebin = 3
        self.cuda_camera_index = 0

    def init_output(self):
        self.data = np.zeros(self.wfs_img_sz * self.wfs_img_sz)
        self.frame_data = np.zeros(self.wfs_img_sz * self.wfs_img_sz, dtype = np.float32)
        self.frame_ctype = self.frame_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


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

        pupil = np.zeros((self.aperture_pix, self.aperture_pix)) + 1
        x = (np.arange(self.aperture_pix) - (self.aperture_pix - 1)/2)
        x = x / self.aperture_pix * self.aperture_size
        pyy, pxx = np.meshgrid(x, x)

        for i in self.used:
            ap = SubAperture()
            ap.position = np.array((xx[i], yy[i]))
            ap.pupil_scal =  self.aperture_size / self.aperture_pix 
            if edge_effect:
                prr = np.sqrt((pxx + xx[i])**2 + (pyy + yy[i])**2)
                ap.pupil = pupil * (prr < (self.whole_pupil_d / 2)).astype(int)
            self.apertures.append(ap)

    def get_data(self, phase):
        self.cu_lib.cuwfs_run(
            self.frame_ctype,
            phase.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            int(self.cuda_camera_index))
        self.data = self.frame_data

    def cuda_destroy(self):
        self.cu_lib.cuwfs_destroy()

    def cuda_plan(self, cu_lib):

        self.cu_lib = cu_lib

        ph_sz = self.phase_pix
        ph_cen = (ph_sz - 1)/2
        sz = self.aperture_pix
        subs = []
        pupils = []

        for ap in self.apertures:
            pos = (ph_cen + ap.position / ap.pupil_scal - sz / 2).astype(int)
            subs.append(pos)
            pupils.append(ap.pupil)

        subs = np.array(subs, dtype=np.int32).flatten()
        pupils = np.array(pupils, dtype=np.int32).flatten()

        edge = (self.plate_pix - self.plate_interval)//2
        bigsz = int(self.plate_interval * self.n_grid + edge * 2)

        big_index = {}
        self.wfs_img_sz = bigsz

        n_bin = self.rebin
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
        ps = (n_sfft - ppx)/2
        ps = int(np.ceil(ps))

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

        patch = a_big_index.astype(np.int32).flatten()

        self.cu_lib.cuwfs_plan(int(n_fft),
            int(self.aperture_pix),
            int(n_sub),
            int(ph_sz),
            int(bigsz),
            int(ni_max),
            subs.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            pupils.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            patch.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))