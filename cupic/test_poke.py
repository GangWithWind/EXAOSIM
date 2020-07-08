# -*- coding: utf-8 -*-
'''
This is a testing program
the program is used to test socket client
'''
import socket
import sys
import struct
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import time

def read_port(file):
    with file.makefile('rb') as fid:
        dim, = struct.unpack('i', fid.read(4))
        shape = struct.unpack('%di'%dim, fid.read(dim*4))
        size, = struct.unpack('i', fid.read(4))
        tp, = struct.unpack('c', fid.read(1))
        n = 1
        for i in range(dim):
            n *= shape[i]
        fmt = '%d%s'%(n, str(tp, encoding="utf-8"))
        data = struct.unpack(fmt, fid.read(n * 4))

    return np.array(data).reshape(shape)


def start_tcp_client():
    # server port and ip
    ip = '127.0.0.1'

    na = 5
    dm = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        dm.connect((ip, 44563))
    except socket.error:
        print('fail to setup socket connection')
        exit()

    wfs = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        wfs.connect((ip, 44561))
    except socket.error:
        print('fail to setup socket connection')
        exit()
    

    vol = np.zeros(na * na, dtype=np.float32)
    for i in range(na * na):
        print(i)
        vol[i] = 5
        vol = vol.astype(np.float32)
        dm.send(vol.tostring())
        vol[i] = 0
        print('send')
        wfs.send(struct.pack("ii", 100, 1))
        print('receive')
        img = read_port(wfs)
        sz = int(np.sqrt(len(img)))
        fits.writeto("./dmfits/wfs_%04d.fits"%i, img.reshape((sz, sz)), overwrite=True)
        time.sleep(0.5)


    vol[0] = -10000
    dm.send(vol.tostring())
    wfs.send(struct.pack("ii", -1, 1))
    dm.close()
    wfs.close()


if __name__ == "__main__":
    start_tcp_client()