import env
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from exaosim.target import Target
from exaosim.control import WFSCal, AOControl, TTMControl
from exaosim.dm import DeformMirror, TipTiltMirror
from exaosim.wfs import WFS
from exaosim.camera import CameraSystem2, CirclePupil


def test_wfs_ctrl():
    img = fits.getdata('_testdata_wfs_image.fits')
    wfs = WFSCal()
    wfs.sub_pix = 5
    wfs.start = [1, 1]
    wfs.n_grid = 32
    wfs.config(img)

def test_ao():
    wfs = WFS()
    wfs.n_grid = 10
    wfs.plate_scale = 0.6
    wfs.plate_interval = 5
    wfs.plate_pix = 15
    wfs.aperture_pix = 10
    wfs.aperture_size = 2 / wfs.n_grid
    wfs.fast_Nbig = 15
    
    wfs.sub_aperture_config()
    wfs.make_sub_aperture(edge_effect=True)

    dm = DeformMirror()
    dm.n_grid = wfs.n_grid + 1
    dm.act_pix = wfs.aperture_pix
    dm.act_config()

    target = Target()

    npix = dm.n_grid * wfs.aperture_pix
    scale = wfs.apertures[0].pupil_scal
    pupil = CirclePupil(wfs.whole_pupil_d, pixsize=npix, scale=scale)
    pupil.make_pupil()
    ccd = CameraSystem2(pupil, plate_scale=0.01, plate_pix=64)
    
    ao = AOControl(wfs, dm, target, ccd)
    ao.init_optics()
    dx = (wfs.plate_pix - wfs.plate_interval)//2
    ao.sub_aperture_config(wfs.n_grid, [dx, dx], wfs.plate_interval)
    ao.poke()

    fits.writeto('poke.fits', ao.poke_mat, overwrite=True)
    ao.ao_on(50)
    fits.writeto('mat.fits', ao.Mat, overwrite=True)


def test_ttm():
    gc = WFS()
    gc.n_grid = 1
    gc.plate_scale = 0.6
    gc.plate_interval = 101
    gc.plate_pix = 101
    gc.aperture_pix = 4 * 10
    gc.aperture_size = 2 / gc.n_grid
    gc.fast_Nbig = 201

    gc.sub_aperture_config()
    gc.make_sub_aperture(edge_effect=True)

    ttm = TipTiltMirror(gc.aperture_pix)
    ttm.n_act = 2

    target = Target()

    ao = TTMControl(gc, ttm, target)
    ao.init_optics()
    guide_size = 41
    start = (gc.plate_pix - guide_size)/2
    ao.guide_cam_config([start, start], guide_size)
    ao.poke()

    ao.ttm_on(50)


def test_poke():
    poke = fits.getdata('poke.fits')
    line = abs(poke).max(axis=1)

    dm = np.zeros((9, 9)).flatten()
    dm[line > line.mean() * 0.5] = 1
    plt.imshow(dm.reshape((9, 9)))
    plt.show()

    # plt.plot(line, '-o')
    # plt.plot([0, 80], [line.mean(), line.mean()])
    # plt.show()

if __name__ == "__main__":
    test_ao()