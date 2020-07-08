import env
import time
import numpy as np
from astropy.io import fits
from cupic.optics import Port, Optics


def simplest_sev():
 
    def processor(ipt):
        return ipt**2

    port = Port()
    port.processor = processor

    port.init_socket()

    time.sleep(60)

opt = Optics()
opt.loop()