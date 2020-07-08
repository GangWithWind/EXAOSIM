# def data_send(socket):
import socket
import struct
import numpy as np
import sys


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


def recv2(sock):
    print("start recv")
    connected = 1
    line = sock.recv(16)
    if len(line) == 0:
        connected = 0
        return np.zeros(1), -1

    pack, ll = struct.unpack("hq", line)
    print('all',ll, 'pack',pack)
    start = 0
    line = bytearray(ll)
    error = []
    len_received = start

    while start < ll:
        # print("recv:", start)
        tmp = sock.recv(pack)
        ltmp = len(tmp)
        if ltmp == 0:
            connected = 0
            break
        line[start:start+ltmp] = tmp
        len_received += ltmp
        start += pack
        start = np.minimum(start, ll)

        if len_received < start:
            # print('rec:',start)
            error.append(start - pack)
        len_received = start

    if not connected:
        np.zeros(1), -1

    error = np.array(error, dtype=np.int64)
    sock.send(struct.pack('i', len(error)))

    print('nerror', len(error))
    print(error)
    if len(error > 0):
        sock.send(error.tostring())

    for e in error:
        
        tmp = sock.recv(pack)
        ltmp = len(tmp)
        print("error recv", e, 'for', ltmp)
        if ltmp == 0:
            connected = 0
            break
        line[e:e+ltmp] = tmp
    if not connected:
        np.zeros(1), -1

    return np.frombuffer(line, dtype=np.float32), 1


def send2(sock, data, pack):
    print("start send")
    line = data.tostring()
    ll = len(line)
    head = struct.pack("hq", pack, ll)
    print(ll, pack, len(head))
    sock.send(head)
    start = 0
    while start < ll:
        sock.send(line[start: start+pack])
        start += pack

    errorline = sock.recv(4)
    if(len(errorline) == 0):
        return -1
    nerror, = struct.unpack("i", errorline)
    print('nerror', nerror)
    errors = []
    if nerror > 0:
        errorline = sock.recv(nerror * 8)
        if len(errorline) == 0:
            return -1
        errors = np.frombuffer(errorline, np.int64)

    for e in errors:
        sock.send(line[e: e+pack])

if __name__ == "__main__":
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    SEND_BUF_SIZE = 1024 * 1024
    RECV_BUF_SIZE = 1024 * 1024
    sock.setsockopt(
        socket.SOL_SOCKET,
        socket.SO_SNDBUF,
        SEND_BUF_SIZE
    )
    sock.setsockopt(
        socket.SOL_SOCKET,
        socket.SO_RCVBUF,
        RECV_BUF_SIZE
    )

    sock.bind(('192.168.1.99', 1234))
    sock.listen(1)
    # cl, addr = sock.accept()
    # data, status = recv2(cl)
    # while True:
    cl, addr = sock.accept()
    ref = np.arange(512 * 512)
    while True:
        print('start recv')
        data, flag = recv3(cl, 512 * 512 * 4)
        
        if flag < 0:
            break
        print(((data - ref)**2).sum())
        data2 = data * 2
        send3(cl, data2)

