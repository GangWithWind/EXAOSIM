#%%
import env
import ctypes
import numpy as np
from cupic.cuwfs import WFS
cu_lib = ctypes.cdll.LoadLibrary("../exaosim/lib/libcuwfs2.so")
import matplotlib.pyplot as plt

#%%
wfs = WFS()
wfs.n_grid = 32
wfs.aperture_pix = 10
wfs.plate_pix = 5
wfs.plate_interval = 5
wfs.rebin = 1
wfs.fast_Nbig = wfs.aperture_pix * 3
wfs.aperture_size = wfs.whole_pupil_d / wfs.n_grid
wfs.phase_pix = wfs.aperture_pix

wfs.sub_aperture_config()
wfs.make_sub_aperture(edge_effect=True)
wfs.cuda_plan(cu_lib)
wfs.init_output()
wfs.cuda_camera_index = 0

wfs.get_data(np.zeros((wfs.phase_pix, wfs.phase_pix)))
plt.imshow(wfs.data.reshape(wfs.wfs_img_sz, wfs.wfs_img_sz))
#%%
ccd = WFS()
ccd.n_grid = 10
ccd.aperture_pix = 10
ccd.plate_pix = 5
ccd.plate_interval = 5
ccd.rebin = 1
ccd.fast_Nbig = ccd.aperture_pix * 3
ccd.aperture_size = ccd.whole_pupil_d / ccd.n_grid
ccd.phase_pix = ccd.aperture_pix

ccd.sub_aperture_config()
ccd.make_sub_aperture(edge_effect=True)
ccd.cuda_plan(cu_lib)
ccd.init_output()
ccd.cuda_camera_index = 1


# %%
ccd.get_data(np.zeros((ccd.phase_pix, ccd.phase_pix)))
plt.imshow(ccd.data.reshape(ccd.wfs_img_sz, ccd.wfs_img_sz))


# %%
