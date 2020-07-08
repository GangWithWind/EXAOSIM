# def data_send(socket):
import socket
import struct
import numpy as np
import sys
from new_sv import *


dmwfs = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
dmwfs.connect(('127.0.0.1', 1234))
send2(dmwfs, np.zeros((1024, 1024), dtype=np.float32), 1024)
wfs = recv2(dmwfs)