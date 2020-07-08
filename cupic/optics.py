import socket
import threading
import numpy as np
import time
import ctypes
import matplotlib.pyplot as plt
from .cuwfs import DM, WFS
from exaosim.aberration import TurbAbrr

cu_lib = ctypes.cdll.LoadLibrary("../exaosim/lib/libcuwfs2.so")


def recv3(sock, ll):
    line = bytearray(ll)
    recved = 0
    while recved < ll:
        tl = sock.recv(ll - recved)
        ltl = len(tl)
        if ltl == 0:
            return np.array(0), -1
        line[recved: recved + ltl] = tl
        recved += ltl
    return np.frombuffer(line, dtype=np.float32), 1


def send3(sock, data):
    line = data.tostring()
    ll = len(line)
    sended = 0
    while sended < ll:
        ls = sock.send(line[sended:])
        sended += ls

class Port(object):
    def __init__(self):
        self.ip = '192.168.1.99'
        # self.ip = '127.0.0.1'
        self.port = 45236
        self.input_size = 100
        self.input_type = np.float32
        self.type_byte = 4
        self.lock = ''
        self.processor = None
        self.s_pack = 1024
        self.n_pack = 1
        self.s_in_pack = 32
        self.n_in_pack = 32

    def init_socket(self):

        self.input_byte = self.input_size * self.type_byte
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        self.sock.setsockopt(
                socket.SOL_SOCKET,
                socket.SO_SNDBUF,
                1024 * 1024)
        self.sock.setsockopt(
                socket.SOL_SOCKET,
                socket.SO_RCVBUF,
                1024 * 1024)

        server_address = (self.ip, self.port)
        self.sock.bind(server_address)
        self.sock.listen(1)
        self.t = threading.Thread(target=self.recv)
        self.t.setDaemon(True)
        self.t.start()

    def recv(self):
        while True:
            print('wait connection..')
            client, addr = self.sock.accept()
            self.client = client
            while True:
                input_data, flag = recv3(client, self.input_size * 4)
                if flag < 0:
                    break
                output_data = self.processor(input_data)
                send3(client, output_data)


class Optics(object):
    def __init__(self):
        dm = DM()
        dm.n_grid = 32
        dm.act_pix = 10
        dm.initial_dm()
        phase_size = dm.npix

        wfs = WFS()
        wfs.n_grid = dm.n_grid - 1
        wfs.aperture_pix = dm.act_pix
        wfs.plate_pix = 15
        wfs.plate_interval = 5
        wfs.rebin = 3
        wfs.fast_Nbig = 15
        wfs.aperture_size = wfs.whole_pupil_d / wfs.n_grid
        wfs.phase_pix = phase_size

        wfs.sub_aperture_config()
        wfs.make_sub_aperture(edge_effect=True)
        wfs.cuda_plan(cu_lib)
        wfs.init_output()
        wfs.cuda_camera_index = 0

        abrr = TurbAbrr()
        abrr.bigsz = [2000, 2000]
        abrr.phase_sz = [phase_size, phase_size]
        abrr.initial()

        dm_wfs_port = Port()
        dm_wfs_port.port = 45236
        dm_wfs_port.input_size = dm.n_act
        dm_wfs_port.processor = self.dm_wfs_fun

        phase_port = Port()
        phase_port.port = 35236
        phase_port.input_size = 1
        phase_port.input_type = np.int32
        phase_port.processor = self.phase_on

        svd_port = Port()
        svd_port.port = 25236
        svd_port.input_size = 1024 * 1498
        svd_port.input_type = np.float32
        svd_port.processor = self.svd_inv_fun()

        self.wfs = wfs
        self.dm = dm
        self.abrr = abrr
        self.dm_wfs_port = dm_wfs_port
        self.phase_port = phase_port
        self.zero_phase = np.zeros(phase_size * phase_size, dtype=np.float32)
        self.phase_on(0)

    def svd_inv_fun(self, poke_mat):        
        dm_eff = (poke_mat**2).sum(axis = 1)
        dm_used = dm_eff > 0.02
        pm_t = poke_mat[dm_used, :]
        U, D, Vt = np.linalg.svd(pm_t, full_matrices=False)
        D2 = 1/(D + 1e-10)
        D2[D < 1e-4] = 0
        Mat = Vt.T @ np.diag(D2) @ U.T
        return Mat.flatten().astype(np.float32)

    def dm_wfs_fun(self, volt):
        self.dm.get_data(volt)
        self.wfs.get_data((self.phase_fun().flatten() + self.dm.phase).astype(np.float32))
        return self.wfs.data

    def phase_on(self, on):
        if on:
            print('abrration on')
            self.phase_fun = self.abrr.get_data_by_time
        else:
            print('laser on')
            self.phase_fun = lambda: self.zero_phase

    def loop(self):
        self.phase_port.init_socket()
        self.dm_wfs_port.init_socket()
        time.sleep(30)
        # plt.ion()
        # f, ax = plt.subplots(2, 2, figsize=(12, 12))
        # for i in range(1000):
        #     time.sleep(0.01)
            # plt.cla()
            # plt.show()
            # ax[0, 0].imshow(self.dm.phase.reshape(self.dm.npix, self.dm.npix))
            # ax[0, 1].imshow(self.wfs.data.reshape(self.wfs.wfs_img_sz, self.wfs.wfs_img_sz))
            # ax[1, 0].imshow(self.abrr.phase.reshape(self.abrr.phase_sz[0], self.abrr.phase_sz[1]))
            # plt.pause(0.01)
            





# class Optics(object):
#     def __init__(self):
#         self.instruments = []
#         self.basic_time = 50 # ms
#         self.total_time = 30 # s
#         self.ps_pix = 320
#         self.phase = np.zeros(self.ps_pix * self.ps_pix, dtype=np.float32)
#         self.ports = []

#     def turbulence(self):
#         pass

#     def main_loop(self):
#         t1 = time.perf_counter() * 1000
#         print("start")
#         for iloop in range(int(total_time * 1000 / frame_time)):
#             t2 = t1 + frame_time


#             while True:
#                 if time.perf_counter()*1000 > t2:
#                     break

#             t1 = t1 + frame_time
        

#     def init_sock(self):
#         pass


#     def init_optics(self, wfs, ccd):
#         self.wfs = wfs
#         self.ccd = ccd


#     def run():
#         pass