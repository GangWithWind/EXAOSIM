import env
import socket
import time
import numpy as np
from astropy.io import fits
from exaosim.control import WFSCal

ip = '127.0.0.1'

target_switch = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    target_switch.connect((ip, 35236))
except socket.error:
    print('fail to setup socket connection')
    exit()

dmwfs = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    dmwfs.connect((ip, 45236))
except socket.error:
    print('fail to setup socket connection')
    exit()

target_switch.send(np.array(0, dtype=np.int32).tostring())

nact = 32 * 32

volt = np.zeros(nact, dtype=np.float32)
nx = 165
npix = 165 * 165

allpoke = []

wc = WFSCal()
wc.start = [5, 5]

dmwfs.send(volt.tostring())
img = []
for ipack in range(nx):
    line = dmwfs.recv(nx * 4)
    data_line = np.frombuffer(line, dtype=np.float32)
    img.append(data_line)
wc.config(np.array(img))

allshifts = []
for i in range(nact):
    print(f'poke {i}')
    volt *= 0
    volt[i] = 1
    dmwfs.send(volt.tostring())

    img = []
    for ipack in range(nx):
        line = dmwfs.recv(nx * 4)
        data_line = np.frombuffer(line, dtype=np.float32)
        img.append(data_line)
    img = np.array(img)
    shifts = wc.shifts_calculation(img)
    allshifts.append(shifts.flatten())

poke_mat = np.array(allshifts)

dm_eff = (poke_mat**2).sum(axis = 1)
dm_used = dm_eff > 0.02
pm_t = poke_mat[dm_used, :]
U, D, Vt = np.linalg.svd(pm_t, full_matrices=False)
D2 = 1/(D + 1e-10)
D2[D < 1e-4] = 0
Mat = Vt.T @ np.diag(D2) @ U.T

target_switch.send(np.array(1, dtype=np.int32).tostring())
time.sleep(1)

print('start ao correction')
volt *= 0
gain = 0.4

for i in range(100):
    print(f'ao loop {i}')
    dmwfs.send(volt.tostring())
    img = []
    for ipack in range(nx):
        line = dmwfs.recv(nx * 4)
        data_line = np.frombuffer(line, dtype=np.float32)
        img.append(data_line)
    img = np.array(img)
    fits.writeto(f'data/aowfs_{i:>04}.fits', img, overwrite=True)
    shifts = wc.shifts_calculation(img).flatten()
    print('RMS:', np.sqrt((shifts**2).sum()))
    volt_out = (shifts @ Mat).astype(np.int32)
    volt[dm_used] -= volt_out * gain

target_switch.close()
dmwfs.close()