import env
import numpy as np
from scipy.special import jv
from exaosim.camera import *
from exaosim.target import *
import matplotlib.pyplot as plt
from astropy.io import fits
pi = np.pi
r2a = 180 * 3600 / pi

def test_run():
    rate = 1
    pupil = Pupil(0.1, 55 * rate, 0.0022 / rate)
    print(500e-9/0.1 / np.pi *180 *3600)
    pupil.make_pupil()
    # pupil.reduce_rate(rate)
    pupil.save_to('p.fits')
    phase = np.zeros((pupil.pixsize, pupil.pixsize))
    ccd = CameraSystem2(pupil, plate_scale=0.1, plate_pix = 128, bin=2)
    target = Target()
    img = ccd.run(phase, target)
    fits.writeto('a.fits', img, overwrite=True)


if __name__ == "__main__":
    test_run()