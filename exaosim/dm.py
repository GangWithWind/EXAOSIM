import numpy as np
from scipy import interpolate

class TipTiltMirror(object):
    def __init__(self, pix):
        self.pix = pix
        x = (np.arange(pix) - (pix - 1)/2)/(pix - 1) * 2
        yy, xx = np.meshgrid(x, x)
        self.yy = yy
        self.xx = xx
        self.n_act = 2
        self.size  = 2

    def run(self, vol):
        return self.xx * vol[0] + self.yy * vol[1]

class DeformMirror(object):
    def __init__(self):
        self.n_grid = 10
        self.act_pix = 20
        self.act_size = 0.1
        self.used = []
        self.all_act = []
        self.error = []
        self.phase_shift_error = 0
        self.vol_coe_mean = 10
        self.vol_coe_error = 0

    def act_config(self):
        # x = np.arange(self.grid_n + 2) + (self.grid_n + 1)/2
        # xx, yy = np.meshgrid()
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

    def run(self, vol):
        self.all_act *= 0
        self.all_act[self.used[0], self.used[1]] = vol * self.vol_coe + self.phase_shift
        npix = self.act_pix * self.n_grid
        x_out = (np.arange(npix) - (npix - 1)/2)/self.act_pix
        sz_in = self.all_act.shape[0]
        x_in = np.arange(sz_in) - (sz_in - 1)/2
        f = interpolate.interp2d(x_in, x_in, self.all_act, kind='cubic')
        return f(x_out, x_out)
