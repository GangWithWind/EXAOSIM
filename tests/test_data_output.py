import env
import ctypes
import numpy as np
from astropy.io import fits
from cupic.cuwfs import WFS
from cupic.cuwfs import DM
cu_lib = ctypes.cdll.LoadLibrary("../exaosim/lib/libcuwfs2.so")

def test_device():

    frame_time = 50  # ms
    total_time = 30  # second

    dm = DM()
    dm.n_grid = 32
    dm.act_pix = 10
    dm.port = 44563
    dm.initial_dm()
    dm.init_socket()
    phase_size = dm.npix

    #initial wfs startcd
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

    vol = np.zeros(dm.n_act)

    vol *= 0
    dm.get_data(vol)
    phase = dm.phase
    phase = phase.astype(np.float32)
    wfs.get_data(phase)
    fits.writeto('data/poke_base.fits', wfs.data.reshape((wfs.wfs_img_sz, wfs.wfs_img_sz)), overwrite=True)

    for i in range(len(vol)):
        vol *= 0
        vol[i] = 1
        dm.get_data(vol)
        phase = dm.phase
        phase = phase.astype(np.float32)
        wfs.get_data(phase)

        fits.writeto(f'data/poke_{i:0>4d}.fits', wfs.data.reshape((wfs.wfs_img_sz, wfs.wfs_img_sz)), overwrite=True)

    for i in range(10):
        vol = np.random.rand(dm.n_act) * 2
        fits.writeto(f'data/volt_{i:0>4d}.fits', vol, overwrite=True)
        dm.get_data(vol)
        phase = dm.phase
        phase = phase.astype(np.float32)
        wfs.get_data(phase)
        fits.writeto(f'data/wfs_{i:0>4d}.fits', wfs.data.reshape((wfs.wfs_img_sz, wfs.wfs_img_sz)), overwrite=True)

    wfs.cuda_destroy()

test_device()