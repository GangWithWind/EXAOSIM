import env
import ctypes
import numpy as np
from scipy.special import jv
from exaosim.camera import *
from exaosim.target import *
import matplotlib.pyplot as plt
from astropy.io import fits
from cupic.cuwfs import WFS
from cupic.cuwfs import DM
from exaosim.port import read_c_array, write_c_array
import time

cu_lib = ctypes.cdll.LoadLibrary("../exaosim/lib/libcuwfs2.so")

def test_dm():
    dm = DM()
    dm.n_grid = 11
    dm.initial_dm()
    dm.receive(None)


def test_wfs_plan():
    wfs = WFS()
    wfs.n_grid = 32
    wfs.aperture_pix = 10
    wfs.paerture_size = 2/wfs.n_grid
    wfs.plate_pix = 15
    wfs.plate_interval = 5
    wfs.rebin = 3
    wfs.fast_Nbig = 15
    wfs.aperture_size = wfs.whole_pupil_d / wfs.n_grid
    
    wfs.sub_aperture_config()
    wfs.make_sub_aperture(edge_effect=True)
    print(wfs.apertures[0].pupil)

    wfs.phase_pix = 320

    wfs.cuda_plan(cu_lib)
    wfs.init_output()

    xx, yy = np.meshgrid(np.arange(wfs.phase_pix), np.arange(wfs.phase_pix))
    phase = np.random.rand(wfs.phase_pix * wfs.phase_pix) * 4
    # phase = xx.flatten()/10 * 2 * np.pi
    wfs.get_data(phase.astype(np.float32))
    fits.writeto('cuwfs.fits', wfs.data.reshape((wfs.wfs_img_sz, wfs.wfs_img_sz)), overwrite=True)


    # imgs = wfs.take_image_fast(phase, target)
    # fits.writeto('wfsf.fits', imgs, overwrite=True)

def plot(ax, wfs_img=None, ccd_img=None, phase=None):

    if np.any(wfs_img):
        self.ax[1, 0].cla()
        self.ax[1, 0].imshow(wfs_img)
        self.ax[1, 0].set_title('wfs image')

    if np.any(ccd_img):
        self.ax[0, 1].cla()
        self.ax[0, 1].imshow(ccd_img)
        self.ax[0, 1].set_title('ccd image')

    if np.any(phase):
        self.ax[1, 1].cla()
        self.ax[1, 1].imshow(phase)
        self.ax[1, 1].set_title('dm shape')



def test_dm_sv():
    frame_time = 100  # ms
    total_time = 30  # second
    phase_size = 320

    dm = DM()
    dm.n_grid = 32
    dm.act_pix = 10
    dm.port = 44563
    dm.initial_dm()
    dm.init_socket()


    t1 = time.perf_counter() * 1000
    print("start")
    t0 = t1

    for iloop in range(int(total_time * 1000 / frame_time)):

        t2 = t1 + frame_time
        phase = np.zeros(phase_size * phase_size, dtype=np.float32)

        phase = phase + dm.phase

        while True:
            if time.perf_counter()*1000 > t2:
                break

        t1 = t1 + frame_time
        # print(time.perf_counter()*1000 - t0)

def test_device():

    frame_time = 50  # ms
    total_time = 30  # second

    dm = DM()
    dm.n_grid = 5
    dm.act_pix = 10
    dm.port = 44563
    dm.initial_dm()
    dm.init_socket()
    phase_size = dm.npix

    #initial wfs start
    wfs = WFS()
    wfs.n_grid = dm.n_grid - 1
    wfs.aperture_pix = 10
    wfs.paerture_size = 2/wfs.n_grid
    wfs.plate_pix = 15
    wfs.plate_interval = 5
    wfs.rebin = 3
    wfs.fast_Nbig = 15
    wfs.aperture_size = wfs.whole_pupil_d / wfs.n_grid
    
    wfs.phase_pix = phase_size
    
    wfs.in_time = frame_time
    wfs.output_interval = 1
    
    wfs.sub_aperture_config()
    wfs.make_sub_aperture(edge_effect=True)
    wfs.cuda_plan(cu_lib)
    wfs.cuda_camera_index = 0
    wfs.init_output()
    wfs.init_socket()

    #initial ccd start
    ccd = WFS()
    ccd.n_grid = 1
    ccd.aperture_pix = wfs.aperture_pix * wfs.n_grid
    ccd.paerture_size = 2/wfs.n_grid
    ccd.plate_pix = 512
    ccd.plate_interval = 512
    ccd.rebin = 1
    ccd.fast_Nbig = 512 * 6
    ccd.aperture_size = wfs.whole_pupil_d / wfs.n_grid
    
    ccd.phase_pix = phase_size
    ccd.in_time = frame_time
    ccd.output_interval = 1
    ccd.port = 46562
    
    ccd.sub_aperture_config()
    ccd.make_sub_aperture(edge_effect=True)
    ccd.cuda_plan(cu_lib)
    ccd.cuda_camera_index = 1
    ccd.init_output()
    ccd.init_socket()

    # f, ax = plt.subplots(2, 2, figsize=(12, 12))

    t1 = time.perf_counter() * 1000
    print("start")
    t0 = t1
    t_plot = t1 / 1000
    for iloop in range(int(total_time * 1000 / frame_time)):

        t2 = t1 + frame_time
        phase = dm.phase
        # print('loop', phase[0:10])
        phase = phase.astype(np.float32)
        # print('fp32', phase[0:10])     
        wfs.receive(phase)
        ccd.receive(phase)

        # if time.perf_counter() > t_plot:
        #     plot(ax, wfs_img=wfs.data.reshape(wfs.wfs_img_sz, wfs.wfs_img_sz), 
        #          ccd_img=ccd.data.reshape(ccd.wfs_img_sz, ccd.wfs_img_sz),
        #          phase=phase.reshape(phase_size, phase_size))
        #     t_plot += 1

        while True:
            if time.perf_counter()*1000 > t2:
                break

        t1 = t1 + frame_time
        # print(time.perf_counter()*1000 - t0)
        
    wfs.cuda_destroy()
    # ccd.cuda_destroy()

if __name__ == "__main__":
    test_device()
