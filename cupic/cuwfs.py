import socket
import threading

from threading import Thread, Lock
import struct
import ctypes
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import interpolate

lock = Lock()


def write_port(fid, data):
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

    fid.write(struct.pack('i', dim))
    for i in range(dim):
        fid.write(struct.pack('i', shape[i]))
    fid.write(struct.pack('i', data.size))
    fid.write(tps)
    fid.write(data.data)


class Device(object):
    def __init__(self):
        self.output_file = 'wfs_img/'
        self.output_interval = 100
        self.output_index = 0
        self.output_inow = 0
        self.output = False

        self.out_frame = 5 #how many frames to outpu set by client
        self.now_frame = 0 
        self.out_stack = 10 #how many frames to stack (intgral time)  set by client
        self.now_stack = 0
        self.in_time = 5 #ms
        self.connect = False
        self.port = 44561
        self.ip = '127.0.0.1'

    def save_output(self):
        if self.output:
            if self.output_inow == self.output_interval:
                self.output_inow = 0
                fits.writeto("%s%5d.fits"%(self.output_file, self.output_index), self.data)
                self.output_index += 1

    def get_data(self, input):
        pass

    def receive(self, input):
        self.get_data(input)
        
        if self.connect:
            self.now_stack += 1

        if self.now_stack == self.out_stack:
            self.now_stack = 0
            if self.connect:
                self.send()
                self.data *= 0
                self.now_frame += 1
                if self.out_frame == self.now_frame:
                    self.connect = False

    def send(self):
        write_port(self.client.makefile('wb'), self.data)
        self.save_output()


    def disconnect(self):
        self.connect = False
        self.now_frame = 0
        self.now_stack = 0
        self.client.close()

    def init_socket(self):

        # SEND_BUF_SIZE = 4096
        # RECV_BUF_SIZE = 4096
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, SEND_BUF_SIZE)
        # self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, RECV_BUF_SIZE)

        # bsize = self.sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
        # print("Buffer size [After] : %d" % bsize)

        server_address = (self.ip, self.port)
        self.sock.bind(server_address)
        self.sock.listen(1)
        self.t = threading.Thread(target=self.control_recv)
        self.t.setDaemon(True)
        self.t.start()

    def control_recv(self):

        while True:
            client, addr = self.sock.accept()
            self.client = client
            while True:
                line = client.recv(10)
                t, n = struct.unpack("ii", line)
                print(t, n)
                if t < 0:
                    # self.sock.close()
                    self.connect = False
                    break
                self.out_frame = n
                self.out_stack = t / self.in_time
                self.data = self.data * 0
                self.connect = True


class DM(Device):
    def __init__(self):
        super().__init__()
        self.n_grid = 10
        self.act_pix = 20
        self.used = []
        self.all_act = []
        self.error = []
        self.phase_shift_error = 0
        self.vol_coe_mean = 1
        self.vol_coe_error = 0
        self.file_i = 0
        

    def receive(self, input):
        return self.phase
        
    def control_recv(self):
        while True:
            client, addr = self.sock.accept()
            self.client = client
            while True:
                line = client.recv(self.n_grid * self.n_grid * 4)
                vol = np.fromstring(line, dtype=np.float32)
                if vol[0] < -1000:
                    break
                self.get_data(vol)
                self.writefile()


    def initial_dm(self):
        acts = np.zeros((self.n_grid + 2, self.n_grid + 2), dtype=float) + 1
        acts[:, 0] = 0
        acts[:, -1] = 0
        acts[0, :] = 0
        acts[-1, :] = 0
        self.n_act = self.n_grid**2
        self.all_act = acts
        self.used = np.where(acts > 0.5)
        self.phase_shift = np.random.randn(self.n_act) * self.phase_shift_error
        self.vol_coe = np.random.randn(self.n_act) * self.vol_coe_error + self.vol_coe_mean
        sz_in = self.all_act.shape[0]
        self.x_in = np.arange(sz_in) - (sz_in - 1)/2
        npix = self.act_pix * self.n_grid
        self.npix = self.act_pix * self.n_grid
        self.x_out = (np.arange(npix) - (npix - 1)/2)/self.act_pix
        self.phase = np.zeros(npix * npix)
        self.vol = np.zeros(npix)
        self.fmt = '%di'%(npix)


    def get_data(self, vol):
        self.all_act[self.used[0], self.used[1]] = vol * self.vol_coe + self.phase_shift
        f = interpolate.interp2d(self.x_in, self.x_in, self.all_act, kind='cubic')
        # lock.acquire()
        # print("\n\n\n\n-----------------------")
        # print('vol', vol, len(vol))
        phase = f(self.x_out, self.x_out).flatten()
        self.phase = phase

        # print('dm phase', self.phase[0:10])
        # lock.release()

    def writefile(self):
        self.file_i += 1
        fits.writeto('./dmfits/phase0_%d.fits'%self.file_i, self.phase.reshape((self.npix, self.npix)), overwrite=True)


class TipTiltMirror(object):
    def __init__(self, pix):
        self.pix = pix
        x = (np.arange(pix) - (pix - 1)/2)/(pix - 1) * 2
        yy, xx = np.meshgrid(x, x)
        self.yy = yy.astype(np.float32).flatten() * 10
        self.xx = xx.astype(np.float32).flatten() * 10
        self.n_act = 2
        self.size = 2
        self.phase = np.zeros(pix * pix)

    def get_data(self, vol):
        self.phase = self.xx * vol[0] + self.yy * vol[1] 


class SubAperture(object):
    def __init__(self):
        self.pos = []
        self.pupil = []


class WFS(Device):
    def __init__(self):
        super().__init__()
        self.n_grid = 10
        self.used = []
        self.aperture_pix = 30
        self.aperture_size = 2 / self.n_grid
        self.wavelength = 500E-9
        self.plate_pix = 21
        self.plate_scale = 0.1
        self.plate_interval = 11
        self.whole_pupil_d = 2
        self.apertures = []
        self.gpu_turbo = True
        self.phase_pix = self.aperture_pix * (self.n_grid + 1)
        self.rebin = 3
        self.cuda_camera_index = 0

    def init_output(self):
        self.data = np.zeros(self.wfs_img_sz * self.wfs_img_sz)
        self.frame_data = np.zeros(self.wfs_img_sz * self.wfs_img_sz, dtype = np.float32)
        self.frame_ctype = self.frame_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


    def sub_aperture_config(self, plot=False):
        x = (np.arange(self.n_grid) - (self.n_grid - 1)/2) * self.aperture_size
        xx, yy = np.meshgrid(x, x)
        rr = np.sqrt(xx * xx + yy * yy).flatten()
        self.used = np.where(rr < (self.whole_pupil_d / 2))[0]

        if plot:
            xx = xx.flatten()
            yy = yy.flatten()
            for i in self.used:
                xc = xx[i]
                yc = yy[i]
                d = self.aperture_size/2
                plt.plot([xc - d, xc - d, xc + d, xc + d, xc - d], 
                [yc - d, yc + d, yc + d, yc - d, yc - d], 'k')
                count = 1000
                xarr=[]
                yarr=[]
                r = self.whole_pupil_d/2
                for i in range(count):
                    j = float(i)/count * 2 * np.pi
                    xarr.append(r*np.cos(j))
                    yarr.append(r*np.sin(j))
                plt.plot(xarr, yarr, 'k')

            plt.show()


    def make_sub_aperture(self, edge_effect=False):
        x = (np.arange(self.n_grid) - (self.n_grid - 1)/2) * self.aperture_size
        xx, yy = np.meshgrid(x, x)
        xx = xx.flatten()
        yy = yy.flatten()

        pupil = np.zeros((self.aperture_pix, self.aperture_pix)) + 1
        x = (np.arange(self.aperture_pix) - (self.aperture_pix - 1)/2)
        x = x / self.aperture_pix * self.aperture_size
        pyy, pxx = np.meshgrid(x, x)

        for i in self.used:
            ap = SubAperture()
            ap.position = np.array((xx[i], yy[i]))
            ap.pupil_scal =  self.aperture_size / self.aperture_pix 
            if edge_effect:
                prr = np.sqrt((pxx + xx[i])**2 + (pyy + yy[i])**2)
                ap.pupil = pupil * (prr < (self.whole_pupil_d / 2)).astype(int)
            self.apertures.append(ap)


    def get_data(self, phase):
        self.cu_lib.cuwfs_run(self.frame_ctype, phase.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 
        int(self.cuda_camera_index))
        self.data = self.frame_data



    def cuda_destroy(self):
        self.cu_lib.cuwfs_destroy()


    def cuda_plan(self, cu_lib):

        self.cu_lib = cu_lib

        ph_sz = self.phase_pix
        ph_cen = (ph_sz - 1)/2
        sz = self.aperture_pix
        subs = []
        pupils = []

        for ap in self.apertures:
            pos = (ph_cen + ap.position / ap.pupil_scal - sz / 2).astype(int)
            subs.append(pos)
            pupils.append(ap.pupil)

        subs = np.array(subs, dtype=np.int32).flatten()
        pupils = np.array(pupils, dtype=np.int32).flatten()

        edge = (self.plate_pix - self.plate_interval)//2
        bigsz = int(self.plate_interval * self.n_grid + edge * 2)

        big_index = {}
        self.wfs_img_sz = bigsz

        n_bin = self.rebin
        n_sub = len(self.apertures)
        n_fft = self.fast_Nbig * n_bin
        
        n_sfft = self.fast_Nbig
        fft_index = np.arange(n_fft * n_fft * n_sub, dtype=int)
        fft_index = fft_index.reshape((n_sub, n_fft, n_fft))

        for i in range(n_sub):
            fft_index[i, :, :] = np.fft.fftshift(fft_index[i, :, :])
        
        fft_index = fft_index.reshape((n_sub, n_sfft, n_bin, n_sfft, n_bin))
        fft_index = np.swapaxes(fft_index, 2, 3)
        fft_index = fft_index.reshape((n_sub, n_sfft, n_sfft, n_bin * n_bin))

        n = self.plate_pix // self.plate_interval + 1

        ppx = self.plate_pix
        ps = (n_sfft - ppx)/2
        ps = int(np.ceil(ps))

        for i, ap in enumerate(self.apertures):
            pos = (np.array(ap.position) / self.aperture_size) * self.plate_interval + (bigsz - self.plate_interval)/2 - edge
            pos = pos.astype(int)

            for y in range(self.plate_pix):
                for x in range(self.plate_pix):
                    b = big_index.get((y + pos[1], x + pos[0]), [])
                    for z in fft_index[i, y + ps, x + ps, :]:
                        b.append(z)
                    big_index[(y + pos[1], x + pos[0])] = b

        ni_max = 0
        for item in big_index.items():
            ni = len(item[1])
            if ni > ni_max:
                ni_max = ni

        a_big_index = np.zeros((bigsz, bigsz, ni_max)) - 1


        for yi, xi in big_index.keys():
            d = big_index[(yi, xi)]
            ni = len(d)
            a_big_index[yi, xi, 0: ni] = np.array(d)

        patch = a_big_index.astype(np.int32).flatten()

        self.cu_lib.cuwfs_plan(int(n_fft),
                               int(self.aperture_pix),
                               int(n_sub),
                               int(ph_sz),
                               int(bigsz),
                               int(ni_max),
                               subs.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                               pupils.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                               patch.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
                               )