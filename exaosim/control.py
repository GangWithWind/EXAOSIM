import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from .zernike import Pzernike_xy, noll_indices


class WFSCal(object):
    def __init__(self):
        self.sub_pix = 5
        self.start = [0, 0]
        self.n_grid = 32
        self.critic = 0.4
    
    def config(self, image):
        stx = self.start[0]
        sty = self.start[1]
        sbp = self.sub_pix
        all_sub = []
        all_sum0 = []
        for iy in range(self.n_grid):
            for ix in range(self.n_grid):
                stx = self.start[0] + ix * self.sub_pix
                sty = self.start[1] + iy * self.sub_pix
                img = image[int(sty): int(sty + sbp), int(stx): int(stx + sbp)]
                all_sub.append((int(stx), int(stx + sbp), int(sty), int(sty + sbp)))
                all_sum0.append(img.sum())

        all_sub = np.array(all_sub)
        all_sum = np.array(all_sum0)
        all_sum.sort()
        all_sum = all_sum[::-1]

        if len(all_sum) > 8:
            value = all_sum[0:len(all_sum)//3].mean()
            used = all_sum0 > (value * self.critic)
        else:
            used = all_sum > -10000

        self.subs = all_sub[used]
        self.n_sub = self.subs.shape[0]
        self.shifts = np.zeros((self.n_sub, 2))

        self.shifts = self.shifts_calculation(image)

    def shifts_calculation(self, image):
        all_shift = []
        for i in range(self.subs.shape[0]):
            s = self.subs[i, :]
            img = image[s[2]: s[3], s[0]: s[1]]
            # sx, sy = self.cenfit(img)
            sx, sy = self.centroid(img)

            all_shift.append((sx, sy))

        return np.array(all_shift) - self.shifts
            

    def cenfit(self, image):
        my, mx = np.unravel_index(np.argmax(image), image.shape)
        sub = np.roll(image, (-my+1, -mx+1), axis=(0,1))
        d = sub.mean(axis=1)
        sy = (d[2] - d[0])/(2 * d[1] - d[0] - d[2])/2# - (img_sz - 1)/2
        d = sub.mean(axis=0)
        sx = (d[2] - d[0])/(2 * d[1] - d[0] - d[2])/2
        img_sp = image.shape
        return mx + sx - (img_sp[1] - 1)/2, my + sy - (img_sp[0] - 1)/2


    def centroid(self, image):
        sp = image.shape
        mx = np.arange(sp[1]) - (sp[1] - 1)/2
        my = np.arange(sp[0]) - (sp[0] - 1)/2
        mxx, myy = np.meshgrid(mx, my)
        sumi = image.sum()
        xs = (mxx * image).sum()/sumi
        ys = (myy * image).sum()/sumi
        return xs, ys


class TTMControl(object):
    def __init__(self, guide, ttm, target):
        self.ttm = ttm
        self.guide = guide
        self.target = target
        wc = WFSCal()
        self.wfs_calulator = wc
        self.gain = 0.3
        self.poke_u = 5
        self.n_act = 2

        plt.ion()
        f, ax = plt.subplots(2, 2, figsize=(12, 12))
        self.ax = ax

    def init_optics(self):
        sz_phase = self.ttm.pix
        self.phase = np.zeros((sz_phase, sz_phase))
        self.phase_ttm = self.phase
        self.phase_dm = self.phase

    def run_ttm(self, vol):
        phase = self.ttm.run(vol)
        self.phase_dm = self.phase + phase

    def run_guide(self):
        img = self.guide.take_image_fast(self.phase_dm, self.target)
        return img

    def guide_cam_config(self, start, size):
        wc = self.wfs_calulator
        wc.sub_pix = size
        wc.start = start
        wc.n_grid = 1

        vol = np.zeros(self.n_act)
        self.run_ttm(vol)
        img = self.run_guide()
        wc.config(img)

    def plot(self, wfs_img=None, ccd_img=None, dm_vol=None, pause=0.01):
        self.ax[0, 0].cla()
        self.ax[0, 0].imshow(self.phase, vmin=-1, vmax=1)
        self.ax[0, 0].set_title('input phase')
        if np.any(wfs_img):
            self.ax[1, 0].cla()
            self.ax[1, 0].imshow(wfs_img)
            self.ax[1, 0].set_title('wfs image')
            for subs in self.wfs_calulator.subs:
                sub = subs - 0.5
                self.ax[1, 0].plot([sub[0], sub[1], sub[1], sub[0], sub[0]], 
                [sub[2], sub[2], sub[3], sub[3], sub[2]], 'r')

        if np.any(ccd_img):
            self.ax[0, 1].cla()
            self.ax[0, 1].imshow(ccd_img)
            self.ax[0, 1].set_title('ccd image')
        if np.any(dm_vol):
            self.ax[1, 1].cla()
            self.ax[1, 1].imshow(self.ttm.run(dm_vol))
            self.ax[1, 1].set_title('dm shape')
        plt.pause(pause)


    def ttm_on(self, n_inter):
        vol0 = np.zeros(self.n_act)
        vol0 = (np.random.rand(self.n_act) - 0.5) * 10
        self.run_ttm(vol0)
        self.phase = self.phase_dm
        sp = self.phase_dm.shape
        vol = vol0 * 0

        for i in range(n_inter):
            img = self.run_guide()
            shifts = self.wfs_calulator.shifts_calculation(img)
            print(i, shifts)
            dvol = shifts.flatten() @ self.Mat * self.gain
            vol += dvol
            # vol = np.maximum(np.minimum(vol, 1), -1)
            self.run_ttm(-vol)
            self.plot(wfs_img=img, dm_vol=vol)

        print('done!')
        plt.ioff()
        plt.show()

    def poke(self):
        vol = np.zeros(self.n_act)
        allshifts = []
        # plt.ion()
        # f, ax = plt.subplots(1, 2, figsize=(12, 6))
        for i in range(self.n_act):
            print('poke', i)
            vol = vol * 0
            vol[i] = self.poke_u
            self.run_ttm(vol)
            img = self.run_guide()

            self.plot(wfs_img=img, dm_vol=vol, pause=2)

            shifts = self.wfs_calulator.shifts_calculation(img)
            allshifts.append(shifts.flatten())
        allshifts = np.array(allshifts)
        
        # fits.writeto('allss.fits', allshifts, overwrite=True)
        # fits.writeto('allpp.fits', np.array(allpos), overwrite=True)
        self.poke_mat = allshifts
        self.matrix_cal(allshifts)

    def matrix_cal(self, allshifts, z_order=-1):
        if z_order < 0:
            U, D, Vt = np.linalg.svd(allshifts, full_matrices=False)
            D2 = 1/(D + 1e-10)
            D2[D < 1e-4] = 0
            self.Mat = Vt.T @ np.diag(D2) @ U.T
            self.Mat = self.Mat * self.poke_u


class AOControl(object):
    def __init__(self, wfs, dm, target, ccd):
        self.dm = dm
        self.wfs = wfs
        self.target = target
        self.n_act = dm.n_act
        wc = WFSCal()
        self.wfs_calulator = wc
        self.gain = 1.1
        self.poke_u = 0.1
        self.ccd = ccd

        plt.ion()
        f, ax = plt.subplots(2, 2, figsize=(12, 12))
        self.ax = ax


    def reshape_phase(self, phase):
        pass

    def plot(self, wfs_img=None, ccd_img=None, dm_vol=None, pause=0.01):
        self.ax[0, 0].cla()
        self.ax[0, 0].imshow(self.phase, vmin=-1, vmax=1)
        self.ax[0, 0].set_title('input phase')
        if np.any(wfs_img):
            self.ax[1, 0].cla()
            self.ax[1, 0].imshow(wfs_img)
            self.ax[1, 0].set_title('wfs image')
            for subs in self.wfs_calulator.subs:
                sub = subs - 0.5
                self.ax[1, 0].plot([sub[0], sub[1], sub[1], sub[0], sub[0]], 
                [sub[2], sub[2], sub[3], sub[3], sub[2]], 'r')

        if np.any(ccd_img):
            self.ax[0, 1].cla()
            self.ax[0, 1].imshow(ccd_img)
            self.ax[0, 1].set_title('ccd image')
        if np.any(dm_vol):
            self.ax[1, 1].cla()
            self.ax[1, 1].imshow(self.dm.run(dm_vol))
            self.ax[1, 1].set_title('dm shape')
        plt.pause(pause)
    
    def init_optics(self):
        sz_phase = self.dm.n_grid * self.dm.act_pix
        self.phase = np.zeros((sz_phase, sz_phase))
        self.phase_ttm = self.phase
        self.phase_dm = self.phase

    def run_dm(self, vol):
        phase = self.dm.run(vol)
        self.phase_dm = self.phase + phase

    def run_wfs(self):
        img = self.wfs.take_image_fast(self.phase_dm, self.target)
        return img

    def run_ccd(self):
        img = self.ccd.run(self.phase_dm, self.target)
        return img

    def sub_aperture_config(self, n_grid, start, size):
        wc = self.wfs_calulator
        wc.sub_pix = size
        wc.start = start
        wc.n_grid = n_grid
        vol = np.zeros(self.n_act)
        self.run_dm(vol)
        img = self.run_wfs()
        wc.config(img)

    def poke(self):
        vol = np.zeros(self.n_act)
        allshifts = []
        for i in range(self.n_act):
            print('poke', i)
            vol = vol * 0
            vol[i] = self.poke_u
            self.run_dm(vol)
            img = self.run_wfs()

            # self.plot(wfs_img=img, dm_vol=vol)

            shifts = self.wfs_calulator.shifts_calculation(img)
            allshifts.append(shifts.flatten())
        allshifts = np.array(allshifts)
        s_mean = abs(allshifts).mean(axis=1)
        self.dm_mask  = np.where(s_mean > (s_mean.mean() * 0.3))[0]

        # fits.writeto('allss.fits', allshifts, overwrite=True)
        # fits.writeto('allpp.fits', np.array(allpos), overwrite=True)
        self.poke_mat = allshifts

        self.matrix_cal(allshifts[self.dm_mask, :])
        

    def matrix_cal(self, allshifts, z_order=-1):
        if z_order < 0:
            U, D, Vt = np.linalg.svd(allshifts, full_matrices=False)
            D2 = 1/(D + 1e-10)
            D2[D < 1e-4] = 0
            self.Mat = Vt.T @ np.diag(D2) @ U.T

    def ao_on(self, n_inter):

        vol0 = np.zeros(self.n_act)
        vol0 = (np.random.rand(self.n_act) - 0.5) * 0.5
        vol1 = vol0.copy()
        vol1[self.dm_mask] = 0
        vol0 = vol0 - vol1
        self.run_dm(vol0)
        self.phase = self.phase_dm
        sp = self.phase_dm.shape
        vol = vol0 * 0

        for i in range(n_inter):
            img = self.run_wfs()
            shifts = self.wfs_calulator.shifts_calculation(img)
            dvol = shifts.flatten() @ self.Mat * self.gain * self.poke_u
            vol[self.dm_mask] += dvol
            vol = vol - vol.mean()
            # vol = np.maximum(np.minimum(vol, 1), -1)
            self.run_dm(-vol)
            ccd_img = self.run_ccd()
            self.plot(wfs_img=img, ccd_img=ccd_img, dm_vol=-vol)
        print('done!')
        plt.ioff()
        plt.show()



# class OpticalSystem(object):
#     def __init__(self):
#         self.instruments = {}
#         self.phases = {}

#     def add_instrment(self, name, obj, strength=1, changer=False):
#         self.instruments[name] = (obj, strength, changer)

#     def get_phase(self, name, bin=1):







