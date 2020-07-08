import socket
import struct
import threading
import time


def wfs_sock(sock):
    while True:
        print('wait for connet')
        client, addr = sock.accept()
        client = client
        while True:
            try:
                print('wait for data..')
                line = client.recv(8)
                if(not len(line)):
                    break
            except socket.error as e:
                print('fail to setup socket connection')
                print(e)
                break

            t, n = struct.unpack("ii", line)

            print(t, n)
            for i in range(n):
                client.send(struct.pack("ii", t, i))
                print('%d send %d'%(t, i))
                time.sleep(0.3)
            


ip = '192.168.1.99'
port = 11561

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

server_address = (ip, port)
sock.bind(server_address)
sock.listen(1)

t = threading.Thread(target=wfs_sock, args=[sock])
t.setDaemon(True)
t.start()
time.sleep(100)


