import numpy as np
import env
from exaosim.utils import *

def test_fft_image():
    
    pupil_image = np.zeros((64, 64))
    pupil_scal = 0.1
    pupil_size = 64
    plate_scal = 0.1

    img = fft_img(pupil_image, pupil_scal, pupil_size, plate_scal)


if __name__ == "__main__":
    test_fft_image()
    