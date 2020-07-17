import env
import numpy as np
from exaosim.aberration import *
import matplotlib.pyplot as plt
from astropy.io import fits

# ab = Aberration()
# ab.initial()
# for i in range(100):
#     print(ab.get_data())


ab = TurbAbrr()
ab.bigsz = np.array([800, 800])
ab.phase_sz = [320, 320]
ab.step = 100
ab.initial()
# for i in range(100):
#     fits.writeto(f'testzab{i:>04}.fits', ab.get_data(), overwrite=True)