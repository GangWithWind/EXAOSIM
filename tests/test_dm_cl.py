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


def start_tcp_client(ip, port):
    # server port and ip
    
    SEND_BUF_SIZE = 4096
    RECV_BUF_SIZE = 4096
    server_ip = ip
    tcp_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # tcp_client.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, SEND_BUF_SIZE)
    # tcp_client.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, RECV_BUF_SIZE)

    # bsize = tcp_client.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
    # print("Buffer size [After] : %d" % bsize)
    try:
        tcp_client.connect((server_ip, port))
    except socket.error:
        print('fail to setup socket connection')
        exit()


    xx, yy = np.meshgrid(np.arange(32), np.arange(32))
    rr = (xx * xx + yy * yy)
    rr = rr.astype(np.int32)
    print(rr.flatten())
    print(len(rr.tostring()))
    tcp_client.send(rr.tostring())
    tcp_client.close()


if __name__ == "__main__":
    start_tcp_client('127.0.0.1', 44563)