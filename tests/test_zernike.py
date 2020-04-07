import env
import numpy as np
from scipy.special import jv
from scipy import interpolate
import matplotlib.pyplot as plt

from exaosim.camera import CameraSystem2 as CameraSystem
from exaosim.target import *

from astropy.io import fits
from exaosim.wfs import WFS
from exaosim.zernike import zernike, Pzernike_xy, noll_indices, ZernikeRBS


def test_img():
    wfs1 = WFS()
    wfs1.n_grid = 5
    wfs1.aperture_size = 2 / wfs1.n_grid
    wfs1.plate_scale = 0.01
    wfs1.plate_interval = 11
    wfs1.plate_pix = 11
    wfs1.used = [7]
    wfs1.make_sub_aperture(edge_effect=False)
    nf = wfs1.n_grid * wfs1.aperture_pix + 100

    j = 4
    n, m = noll_indices(j)
    phase = zernike(n, m, npix=nf, outside=0)

    fits.writeto('phase.fits', phase, overwrite=True)

    target = Target()
    alls = []

    imgs1, shifts1, sss = wfs1.take_image(phase, target, shifts=True, slope_shifts=True)
    fits.writeto('imgs1.fits', imgs1, overwrite=True)
    index = 0
    ap = wfs1.apertures[index]
    ap_position = np.array(ap.position)

    x = (np.array(ap.position) / ap.pupil_scal)/(nf - 1) *2
    zx, zy = Pzernike_xy(n, m, [x[1]], [x[0]])
    zpd = ZernikeRBS(phase, pix=False)
    x_r, y_r = zpd.get(x[0], x[1])

    factor = ap.wavelength / (np.pi / 180 / 3600) / 2 / np.pi / ap.pupil_scal

    print('slope_r', x_r/(nf-1)*2, y_r/(nf-1)*2)
    print('slope_i', zx/(nf-1)*2, zy/(nf-1)*2)
    print('shift from image', shifts1[index][0]*wfs1.plate_scale,shifts1[index][1]*wfs1.plate_scale, '+-', wfs1.plate_scale)
    print('shift from slope', zx[0] /(nf-1)*2 * factor, zy[0] /(nf-1)*2 * factor)
    print('shift from phase', sss[index][1], sss[index][0])
    

def test_zernike_pd():
    j = 22
    n, m = noll_indices(j)
    nf = 513
    phase = zernike(n, m, npix=nf, outside=0)
    zx, zy = Pzernike(n, m, npix=nf, outside=0)
    zpd = ZernikeRBS(phase, pix=False)
    sz = phase.shape[0]
    xx = (np.arange(sz) - (sz - 1)/2)/(sz/2 - 0.5)

    ix = (nf - 1)//2 + 20
    pzx = []
    pzxi = []
    pzy = []
    pzyi = []
    for iy in range(20, 400):
        pzx.append(zx[ix, iy])
        pzy.append(zy[ix, iy])
        yi, xi = zpd.get(xx[ix], xx[iy])
        pzxi.append(xi)
        pzyi.append(yi)
    plt.plot(pzx, 'C1-', label = 'x')
    plt.plot(pzy, 'C2-', label = 'y')
    plt.plot(pzxi, 'C1x', label='xi')
    plt.plot(pzyi, 'C2x', label='yi')
    plt.show()
    
def test_wfs_zernike():
    wfs = WFS()

    wfs.n_grid = 5
    wfs.plate_scale = 0.04
    wfs.plate_interval = 11
    wfs.plate_pix = 11
    wfs.aperture_pix = 20
    wfs.aperture_size = 2 / wfs.n_grid

    wfs.sub_aperture_config()
    wfs.make_sub_aperture(edge_effect=False)
    nf = wfs.n_grid * wfs.aperture_pix + 20
    target = Target()

    j = 2
    n, m = noll_indices(j)
    phase = zernike(n, m, npix=nf, outside=0)

    imgs, shifts = wfs.take_image(phase, target, shifts=True)
    print(shifts)
    fits.writeto('z_img.fits', imgs, overwrite=True)
    ap_position = []

    for ap in wfs.apertures:
        ap_position.append(ap.position)

    x = (np.array(ap_position) / ap.pupil_scal)/(nf - 1) *2

    zx, zy = Pzernike_xy(n, m, x[:, 1], x[:, 0])

    ap = wfs.apertures[0]

    shifts = np.array(shifts)

    factor = ap.wavelength / (np.pi / 180 / 3600) / 2 / np.pi / ap.pupil_scal
    print(wfs.plate_scale)


    plt.plot(shifts[:, 0] * wfs.plate_scale, 'C1o', label = 'wfs_x')
    plt.plot(shifts[:, 1] * wfs.plate_scale, 'C2o', label = 'wfs_y')
    plt.plot(zx * factor / (nf-1)*2, 'C1x-', label = 'pz_x')
    plt.plot(zy * factor / (nf-1)*2, 'C2x-', label = 'pz_y')
    plt.legend()
    plt.show()

    fits.writeto('wfs_z_10.fits', imgs, overwrite=True)


def test_wfs_reverse():
    wfs = WFS()
    wfs.aperture_pix = 15
    wfs.n_grid = 10
    wfs.plate_scale = 0.1
    wfs.plate_interval = 5
    wfs.plate_pix = 7
    wfs.aperture_size = 2 / wfs.n_grid

    wfs.sub_aperture_config()
    wfs.make_sub_aperture(edge_effect=False)
    nf = int(wfs.whole_pupil_d / wfs.apertures[0].pupil_scal) + 100
    print(nf)
    nf = wfs.n_grid * wfs.aperture_pix +100
    print(nf)
    target = Target()
    
    print(len(wfs.apertures))
    jn = 30
    alpha = np.random.rand(jn)
    phase = 0
    for j in range(jn):
        n, m = noll_indices(j+2)
        phase_j = zernike(n, m, npix=nf, outside=0)
        phase = phase + phase_j * alpha[j]

    fits.writeto('ps.fits', phase, overwrite=True)

    imgs, shifts = wfs.take_image(phase, target, shifts=True)
    fits.writeto('z_img_alpha.fits', imgs, overwrite=True)
    fits.writeto('z_alpha.fits', alpha, overwrite=True)
    ap_position = []

    for ap in wfs.apertures:
        ap_position.append(ap.position)

    x = (np.array(ap_position) / ap.pupil_scal)/(nf - 1) * 2

    mat = []
    for j in range(jn):
        n, m = noll_indices(j+2)
        zx, zy = Pzernike_xy(n, m, x[:,1], x[:,0])
        zxy = np.array([zx, zy]).flatten()
        mat.append(zxy)

    ap = wfs.apertures[0]
    factor = ap.wavelength / (np.pi / 180 / 3600) / 2 / np.pi / ap.pupil_scal
    mat = np.array(mat) * factor / (nf - 1) * 2

    shifts = np.array(shifts).T.flatten() * wfs.plate_scale

    f, ax = plt.subplots(1,2,figsize = (12,6))
    ax[0].plot(shifts, '-o', label="shifts")
    ax[0].plot(alpha @ mat, label="calcualted")
    ax[0].legend()

    ax[1].plot(alpha, '-o', label="alpha")
    ax[1].plot(shifts @ mat.T @ np.linalg.inv(mat @ mat.T), label="calcualted")
    ax[1].legend()

    plt.show()


if __name__ == "__main__":
    # test_img()
    # test_wfs_zernike()
    test_wfs_reverse()
