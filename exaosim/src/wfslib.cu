#include <stdio.h>
#include <cufft.h>
#include <cuda_runtime.h>
#include "../inc/g_array.h"

extern "C" void cuwfs_plan(int n_big, int sub_sz, int nsub, int phase_sz, int wfs_sz, int patcah_sz, 
    int* subs, int* pupil, int* patch);
extern "C" void cuwfs_run(float* wfs_img, float* phase);
extern "C" void cuwfs_destroy();

cufftHandle fftplanfwd;
int* d_subs;
int* d_patch;
int* d_pupil;

cufftComplex *d_fft_input;
cufftComplex *d_fft_output;
float* d_wfs;
float* d_phase;

int g_n_big;
int g_sub_sz;
int g_nsub;
int g_phase_sz;
int g_wfs_sz;
int g_patch_sz;

int threadsPerBlock;
int blocksPerGrid;


__global__ void SetFFTInput(cufftComplex* output, float* phase, int* subs, int *pupil, 
    int H, int W, int Hs, int Ws, int Wp, int n_subs){
    
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i < H * n_subs){
        int isub = i / H;
        int ix = i % H;

        int is0 = subs[isub * 2 + 1];
        int js0 = subs[isub * 2];

        if(ix < Hs){
            int j0 = (is0 + ix) * Wp + js0;
            int jp = i * W;
            for(int j = 0; j < Ws; j++){
                float value = phase[j0 + j];
                output[jp + j].x = cos(value) * pupil[isub * Hs * Ws + ix * Ws + j];
                output[jp + j].y = sin(value) * pupil[isub * Hs * Ws + ix * Ws + j];;
            }
        }
    }
}


__global__ void FFTpatch(float* output, cufftComplex* fftres, int* patch, 
    int n_big, int pz){
    
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i < n_big * n_big){
        int fft_index;
        output[i] = 0;
        
        for(int j = 0; j < pz; j++){
            fft_index = patch[i * pz + j];
            if(fft_index >= 0){
                cufftComplex value = fftres[fft_index];
                output[i] += value.x * value.x + value.y * value.y;
            }
        }
    }

}


void cuwfs_plan(int n_big, int sub_sz, int nsub, int phase_sz, int wfs_sz, 
    int patch_sz, int* subs, int* pupil, int* patch){
    //initial device memory

    cudaMalloc((void**)&d_subs, 2 * nsub * sizeof(int));
    cudaMemcpy(d_subs, subs, 2 * nsub * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_patch, wfs_sz * wfs_sz * patch_sz * sizeof(int));
    cudaMemcpy(d_patch, patch, wfs_sz * wfs_sz * patch_sz * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_pupil, sub_sz * sub_sz * nsub* sizeof(int));
    cudaMemcpy(d_pupil, pupil, sub_sz * sub_sz * nsub* sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_fft_input, n_big * n_big * nsub * sizeof(cufftComplex));
    cudaMemset(d_fft_input, 0, n_big * n_big * nsub * sizeof(cufftComplex));

    cudaMalloc((void**)&d_fft_output, n_big * n_big * nsub * sizeof(cufftComplex));
    cudaMalloc((void**)&d_wfs, wfs_sz * wfs_sz * sizeof(float));
    cudaMalloc((void**)&d_phase, phase_sz * phase_sz * sizeof(float));

    g_n_big = n_big;
    g_sub_sz = sub_sz;
    g_nsub = nsub;
    g_phase_sz = phase_sz;
    g_wfs_sz = wfs_sz;
    g_patch_sz = patch_sz;

    threadsPerBlock = 128;
    blocksPerGrid = (n_big * nsub + threadsPerBlock - 1) / threadsPerBlock;

    int H = n_big;
    int W = n_big;
    const int rank = 2;//维数
    int n[rank] = {H, W};//n*m
    int*inembed = n;//输入的数组sizecudaMemcpyHostToDevice
    int istride = 1;//数组内数据连续，为1
    int idist = n[0] * n[1];//1个数组的内存大小
    int*onembed = n;//输出是一个数组的size
    int ostride = 1;//每点DFT后数据连续则为1
    int odist = n[0] * n[1];//输出第一个数组与第二个数组的距离，即两个数组的首元素的距离
    int batch = nsub;//批量处理的批数

    cufftPlanMany(&fftplanfwd, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, batch);
}

void cuwfs_run(float* wfs_img, float* phase){

    cudaMemcpy(d_phase, phase, g_phase_sz * g_phase_sz * sizeof(float), cudaMemcpyHostToDevice);
    SetFFTInput<<<blocksPerGrid, threadsPerBlock>>>(d_fft_input, d_phase, d_subs, d_pupil,
        g_n_big, g_n_big, g_sub_sz, g_sub_sz, g_phase_sz, g_nsub);
    cudaDeviceSynchronize();

    cufftExecC2C(fftplanfwd, d_fft_input, d_fft_output, CUFFT_FORWARD);
    cudaDeviceSynchronize();

    cufftComplex* fft_output = (cufftComplex*)malloc(g_nsub * g_n_big * g_n_big * sizeof(cufftComplex));
    ARRAY *fft_res = array_zeros(3, g_nsub, g_n_big, g_n_big);
    cudaMemcpy(fft_output, d_fft_output, g_nsub * g_n_big * g_n_big * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

    for(int i = 0; i < fft_res->size; i++){
        fft_res->data[i] = fft_output[i].x * fft_output[i].x + fft_output[i].y * fft_output[i].y;
    }

    FILE *fp;
    if(!(fp = fopen("middle.bin", "wb"))){
        printf("array file error!");
    }
    array_save(fp, fft_res);
    fclose(fp);



    blocksPerGrid = (g_wfs_sz * g_wfs_sz + threadsPerBlock - 1) / threadsPerBlock;
    
    FFTpatch<<<blocksPerGrid, threadsPerBlock>>>(d_wfs, d_fft_output, d_patch, 
        g_wfs_sz, g_patch_sz);

    cudaDeviceSynchronize();

    cudaMemcpy(wfs_img, d_wfs, g_wfs_sz * g_wfs_sz * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

}


void cuwfs_destroy(){
    cudaFree(d_subs);
    cudaFree(d_patch);
    cudaFree(d_fft_output);
    cudaFree(d_fft_input);
    cudaFree(d_wfs);
    cudaFree(d_pupil);
    cudaFree(d_phase);
    cufftDestroy(fftplanfwd);
}