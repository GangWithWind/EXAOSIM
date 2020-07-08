import env
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from exaosim.target import Target
from exaosim.control import WFSCal, AOControl, TTMControl
from exaosim.dm import DeformMirror, TipTiltMirror
from exaosim.wfs import WFS
from exaosim.camera import CameraSystem2, CirclePupil
from exaosim.port import read_c_array, write_c_array


def make_ttm_data():
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

    write_c_array('testdata/ttm_poke_mat.bin', ao.poke_mat)
    write_c_array('testdata/ttm_trans_mat.bin', ao.Mat)
    write_c_array('testdata/ttm_shifts0.bin', ao.wfs_calulator.shifts)
    write_c_array('testdata/ttm_config.bin', ao.wfs_calulator.subs.astype(float))

    vol0 = np.zeros(ao.n_act)
    vol0 = (np.random.rand(ao.n_act) - 0.5) * 10
    ao.run_ttm(vol0)
    ao.phase = ao.phase_dm

    img = ao.run_guide()
    write_c_array('testdata/ttm_image.bin', img)
    shifts = ao.wfs_calulator.shifts_calculation(img)
    write_c_array('testdata/ttm_shifts.bin', shifts)

    dvol = shifts.flatten() @ ao.Mat
    write_c_array('testdata/ttm_voltage.bin', dvol)


def make_wfs_data():
    wfs = WFS()
    wfs.n_grid = 31
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

    subs = []
    ph_sz = dm.n_grid * dm.act_pix
    ph_cen = (ph_sz - 1)/2
    sz = wfs.aperture_pix

    pupils = []

    for ap in wfs.apertures:
        pos = (ph_cen + ap.position / ap.pupil_scal - sz / 2).astype(int)
        subs.append(pos)
        pupils.append(ap.pupil.pupil)

    subs = np.array(subs)
    pupils = np.array(pupils)
    
    write_c_array('testdata/wfs_subs.bin', subs)
    write_c_array('testdata/wfs_pupils.bin', pupils)
    
    target = Target()

    npix = wfs.n_grid * wfs.aperture_pix
    scale = wfs.apertures[0].pupil_scal
    pupil = CirclePupil(wfs.whole_pupil_d, pixsize=npix, scale=scale)
    pupil.make_pupil()
    ccd = CameraSystem2(pupil, plate_scale=0.01, plate_pix=64)
    
    ao = AOControl(wfs, dm, target, ccd)
    ao.init_optics()
    dx = (wfs.plate_pix - wfs.plate_interval)//2
    ao.sub_aperture_config(wfs.n_grid, [dx, dx], wfs.plate_interval)
    
    ao.poke()

    write_c_array('testdata/poke_mat.bin', ao.poke_mat)
    write_c_array('testdata/trans_mat.bin', ao.Mat)
    write_c_array('testdata/dm_config.bin', ao.dm_mask.astype(float))
    write_c_array('testdata/shifts0.bin', ao.wfs_calulator.shifts)
    write_c_array('testdata/wfs_config.bin', ao.wfs_calulator.subs.astype(float))

    vol0 = np.zeros(ao.n_act)
    vol0[ao.dm_mask] = (np.random.rand(ao.Mat.shape[1])-0.5) * 0.5
    ao.run_dm(vol0)
    ao.phase = ao.phase_dm
    write_c_array('testdata/wfs_phase.bin', ao.phase)
    img = ao.run_wfs()
    write_c_array('testdata/wfs_image.bin', img)
    shifts = ao.wfs_calulator.shifts_calculation(img)
    write_c_array('testdata/shift.bin', shifts)

    dvol = shifts.flatten() @ ao.Mat
    vol = np.zeros(ao.n_act)
    vol[ao.dm_mask] = dvol
    write_c_array('testdata/voltage.bin', vol)


if __name__ == "__main__":
    make_wfs_data()