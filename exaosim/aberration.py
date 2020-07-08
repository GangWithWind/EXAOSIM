import time
import numpy as np
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
        self.speed = 10


    def initial(self):
        xx, yy = np.meshgrid(np.arange(self.bigsz[0]), np.arange(self.bigsz[1]))
        self.big_phase = xx# * xx + yy * yy #np.zeros(self.bigsz)

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
        self.r0 = 0.1
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
        self.big_phase = self.new_phs()
        self.phase = np.zeros(self.phase_sz)
