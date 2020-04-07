import env
import numpy as np
from scipy.special import jv
from exaosim.camera import *
from exaosim.target import *
import matplotlib.pyplot as plt
from astropy.io import fits
from exaosim.wfs import WFS

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

if __name__ == "__main__":
    test_fast_wfs()