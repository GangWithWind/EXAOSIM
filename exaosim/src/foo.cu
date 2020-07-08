#include <cuda_runtime.h>
#include <stdlib.h> 
#include <stdio.h>

int global_i = 2;

extern "C" int foo(int);
extern "C" int get_global();
extern "C" int set_global(int);

extern "C" int array_trans_i(int* array, int n);
extern "C" int array_trans_l(long* array, int n);
extern "C" int array_trans_f(float* array, int n);

int get_global(){
    return global_i;
}

int array_trans_i(int* array, int n){
    for(int i = 0; i < n; i++){
        printf("%d\n", array[i]);
        array[i] = array[i] * 2;
    }
    return 0;
}

int array_trans_l(long* array, int n){
    for(int i = 0; i < n; i++){
        printf("%ld\n", array[i]);
        array[i] = array[i] * 2;
    }
    return 0;
}

int array_trans_f(float* array, int n){
    for(int i = 0; i < n; i++){
        printf("%.1f\n", array[i]);
        array[i] = array[i] * 2;
    }
    return 0;
}

int set_global(int i){
    global_i = i;
    return (int) 11;
}

__global__ void gpu(float *A, float *B, int N){

    int ib = blockDim.x * blockIdx.x + threadIdx.x;
    if (ib < N){
        B[ib] = A[ib] * A[ib];
    }
}

int foo(int a){
    int N = 10;
    float A[N];
    float B[N];

    for(int i = 0; i < N; i++){
        A[i] = a + i;
    }

    int threadsPerBlock = 20;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    float *GA, *GB;
    cudaMalloc((void**)&GA, N * sizeof(float));
    cudaMalloc((void**)&GB, N * sizeof(float));

    cudaMemcpy(GA, A, N * sizeof(float), cudaMemcpyHostToDevice);
    gpu<<<blocksPerGrid, threadsPerBlock>>>(GA, GB, N);
    cudaMemcpy(B, GB, N * sizeof(float), cudaMemcpyDeviceToHost);

    float sum = 0;
    for(int i = 0; i < N; i++){
        sum += B[i];
    }

    cudaFree(A);

    printf("from cuda");
    return (int) sum;
}