#include <stdio.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <netdb.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <stdio.h>
#include "/home/gzhao/Work/CSTP/concept/inc/array.h"

int array_sock_send(int sock, ARRAY * array, int pack_size){
    struct Head{
        short pack_size;
        long length;
    } head;
    head.length = array->size * 4;
    head.pack_size = pack_size;
    long len = head.length;
    send(sock, (char*)&head, sizeof(head), 0);
    printf("length:%ld, size:%d, %ld\n", head.length, head.pack_size, sizeof(head));

    long start = 0;
    int ss = head.pack_size;
    while(start < len){
        // printf("send %ld\n", start);
        if(start + ss > len){
            ss = len - start;
        }
        send(sock, (char*)(array->data) + start, ss, 0);
        start += ss;
    }
    int n_error;
    recv(sock, &n_error, sizeof(int), 0);
    printf("nerror: %d\n", n_error);
    long errors[10];

    if(n_error > 0 && n_error < 10){
        recv(sock, errors, n_error * sizeof(start), 0);
    }else if(n_error > 10){
        return -1;
    }

    ss = head.pack_size;
    for(int i = 0; i < n_error; i++){
        start = errors[i];
        if(start + ss > len){
            ss = len - start;
        }
        send(sock, (char*)(array->data) + errors[i], ss, 0);
    }
}


int array_sock_recv(int sock, ARRAY *array){
    struct Head{
        short pack_size;
        long length;
    } head;
    recv(sock, &head, sizeof(head), 0);
    printf("length:%ld, size:%d, %ld\n", head.length, head.pack_size, sizeof(head));
    long start = 0;
    int ss = head.pack_size;
    int recv_size;
    int n_error = 0;
    long errors[10];

    while(start < head.length){
        if(start + ss > head.length){
            ss = head.length - start;
        }
        // printf("recv %ld for %d\n", start, ss);
        recv_size = recv(sock, (char*)array->data + start, ss, 0);
        if(recv_size != ss){
            printf("error! start:%ld\n",start);
            errors[n_error] = start;
            n_error ++;
        }
        if(n_error > 10){
            return -1;
        }
        start = start + ss;
    }
    printf("nerror %d, sizeof %d\n", n_error, (int)sizeof(n_error));
    send(sock, (char*)&n_error, sizeof(n_error), 0);
    for(int i = 0; i < n_error; i++){
        send(sock, (char*)(errors + i), sizeof(errors[i]), 0);
    }

    for(int i = 0; i < n_error; i++){
        start = errors[i];
        ss = head.pack_size;
        if(ss + start > head.length){
            ss = head.length - start;
        }
        recv(sock, (char*)array->data + start, ss, 0);
    }
}

int link_to_service(const char* ip, int port){
    int sock;
    struct sockaddr_in abrrAddr;
	if((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0)
	{
		perror("socket");
		return -1;
	}

	abrrAddr.sin_family = AF_INET;
	abrrAddr.sin_port = htons(port);

	abrrAddr.sin_addr.s_addr = inet_addr(ip);
	if(connect(sock, (struct sockaddr *)&abrrAddr, sizeof(abrrAddr)) < 0)
	{
		perror("connect");
		return -1;
	}
    return sock;
}

int main(){
    int abrrSock =link_to_service("127.0.0.1", 1234);
    ARRAY *dm = array_zeros(2, 2048, 2048);
    ARRAY *wfs = array_zeros(2, 2048, 2048);
    for(int i = 0; i < 100; i++){
        printf("start send\n");
        array_sock_send(abrrSock, dm, 1024);
        printf("start recv\n");
        array_sock_recv(abrrSock, wfs);
    }

}