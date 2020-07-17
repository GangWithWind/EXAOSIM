import env
import time
import numpy as np
from astropy.io import fits
from aoserver.optics import Optics

opt = Optics()
opt.loop()