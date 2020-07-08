import env
import numpy as np
from scipy.special import jv
from exaosim.camera import *
from exaosim.target import *
import matplotlib.pyplot as plt
from astropy.io import fits
from exaosim.wfs import WFS
from exaosim.port import read_c_array, write_c_array
import time

def test_wfs():
    wfs = WFS()
    wfs.n_grid = 10
    wfs.plate_pix = 7
    wfs.plate_interval = 5
    wfs.plate_scale = 0.2
    wfs.aperture_size = wfs.whole_pupil_d / wfs.n_grid

    wfs.sub_aperture_config()
    wfs.make_sub_aperture(edge_effect=True)
    nf = wfs.n_grid * wfs.aperture_pix + 2
    phase = np.zeros((nf, nf))
    target = Target()
    imgs = wfs.take_image(phase, target)
    fits.writeto('wfsi.fits', imgs, overwrite=True)

def test_fast_wfs():
    wfs = WFS()
    wfs.n_grid = 32
    wfs.aperture_pix = 10
    wfs.plate_pix = 7
    wfs.plate_interval = 5
    wfs.plate_scale = 0.2
    wfs.fast_Nbig = 31
    wfs.aperture_size = wfs.whole_pupil_d / wfs.n_grid
    
    wfs.sub_aperture_config()
    wfs.make_sub_aperture(edge_effect=True)
    nf = wfs.n_grid * wfs.aperture_pix + 2
    phase = np.zeros((nf, nf))
    target = Target()
    imgs = wfs.take_image_fast(phase, target)
    fits.writeto('wfsf.fits', imgs, overwrite=True)


def test_wfs_plan_old():
    wfs = WFS()
    wfs.n_grid = 32
    wfs.aperture_pix = 10
    wfs.plate_pix = 15
    wfs.plate_interval = 5
    wfs.plate_scale = 0.6
    wfs.fast_Nbig = 15
    wfs.aperture_size = wfs.whole_pupil_d / wfs.n_grid
    
    wfs.sub_aperture_config()
    wfs.make_sub_aperture(edge_effect=True)
    nf = wfs.n_grid * wfs.aperture_pix + 2
    phase = np.zeros((nf, nf))
    target = Target()
    wfs.cuda_plan()

def test_wfs_plan():
    wfs = WFS()
    wfs.n_grid = 1
    wfs.aperture_pix = 50
    wfs.plate_pix = 201
    wfs.plate_interval = 201
    wfs.fast_Nbig = 1001
    wfs.aperture_size = wfs.whole_pupil_d / wfs.n_grid
    
    wfs.sub_aperture_config()
    wfs.make_sub_aperture(edge_effect=True)

    wfs.cuda_plan()

    phase = read_c_array('../testdata/wfs_phase.bin')
    phase *= 0

    t1 = time.perf_counter()

    for i in range(1):
        res = wfs.cuda_run(phase)
        
    res2 = wfs.take_image_fast(phase, Target())

    t2 = time.perf_counter()
    print(t2 - t1)
    fits.writeto('old.fits', res, overwrite=True)


    # imgs = wfs.take_image_fast(phase, target)
    # fits.writeto('wfsf.fits', imgs, overwrite=True)




if __name__ == "__main__":
    test_wfs_plan()
