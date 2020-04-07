import env
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from exaosim.dm import DeformMirror
from exaosim.wfs import WFS
from exaosim.target import Target


def test_dm_unit():
    dm = DeformMirror()
    dm.act_config()
    vol = np.random.rand(dm.n_grid**2)
    plt.ion()
    for i in range(len(vol)):
        vol *= 0
        vol[i] = 1
        img = dm.run(vol)
        # plt.cla()
        # plt.imshow(img)
        # plt.pause(0.1)
    # fits.writeto('dm_out.fits', img, overwrite=True)

def test_dm_wfs():
    wfs = WFS()
    wfs.n_grid = 32
    wfs.plate_scale = 0.6
    wfs.plate_interval = 5
    wfs.plate_pix = 5
    wfs.aperture_pix = 20
    wfs.aperture_size = 2 / wfs.n_grid
    wfs.sub_aperture_config()
    wfs.make_sub_aperture(edge_effect=True)

    dm = DeformMirror()
    dm.n_grid = wfs.n_grid + 1
    dm.act_pix = wfs.aperture_pix
    dm.act_config()

    target = Target()

    vol = np.random.rand(dm.n_grid**2)
    phase = dm.run(vol)
    
    img = wfs.take_image(phase, target)

    fits.writeto('dm_wfs.fits', img, overwrite = True)



if __name__ == "__main__":
    # test_dm_unit()
    test_dm_wfs()