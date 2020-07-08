import struct
import numpy as np
import time


def read_c_array(file):
    with open(file, 'rb') as fid:
        dim, = struct.unpack('i', fid.read(4))
        shape = struct.unpack('%di'%dim, fid.read(dim*4))
        size, = struct.unpack('i', fid.read(4))
        tp, = struct.unpack('c', fid.read(1))
        print('dim', dim)
        print('shape', shape)
        print('size', size)
        print('tp', tp)
        n = 1
        for i in range(dim):
            n *= shape[i]
        fmt = '%d%s'%(n, str(tp, encoding="utf-8"))
        data = struct.unpack(fmt, fid.read(n * 4))

    return np.array(data).reshape(shape)


def write_c_array(file, data):
    shape = data.shape
    dim = len(shape)
    n = 1
    for i in range(dim):
        n *= shape[i]

    if (data.dtype == np.float64 or data.dtype == np.float32):
        data = data.astype(np.float32)
        tps = b'f'

    if (data.dtype == np.int64 or data.dtype == np.int32):
        data = data.astype(np.int32)
        tps = b'i'

    with open(file, 'wb') as fid:
        fid.write(struct.pack('i', dim))
        for i in range(dim):
            fid.write(struct.pack('i', shape[i]))
        print(data.size)
        fid.write(struct.pack('i', data.size))
        fid.write(tps)
        fid.write(data.data)