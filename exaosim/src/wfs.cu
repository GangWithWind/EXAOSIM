#include <stdio.h>
#include <cufft.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "../inc/g_array.h"

cufftHandle fftplanfwd;

__global__ void SetFFTInput(cufftComplex* output, float* phase, float* subs, 
    int H, int W, int Hs, int Ws, int Wp, int n_subs){
    
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i < H * n_subs){
        int isub = i / H;
        int ix = i % H;

        int is0 = subs[isub * 2];
        int js0 = subs[isub * 2 + 1];

        if(ix < Hs){
            int j0 = (is0 + ix) * Wp + js0;
            int jp = i * W;
            for(int j = 0; j < Ws; j++){
                float value = phase[j0 + j];
                output[jp + j].x = cos(value);
                output[jp + j].y = sin(value);
            }
        }
    }
}


__global__ void FFTpatch(float* output, cufftComplex* fftres, float* patch, 
    int n_big, int pz){
    
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i < n_big * n_big){
        long fft_index;
        
        for(int j = 0; j < pz; j++){
            fft_index = patch[i * pz + j];
            if(fft_index > 0){
                cufftComplex value = fftres[fft_index];
                output[i] += value.x * value.x + value.y * value.y;
            }
        }   
    }

}

int main(void){

    float time_elapsed = 0;

    ARRAY* phase = array_load_file("../testdata/wfs_phase.bin");
    array2device(phase);
    ARRAY* wfs = array_load_file("../testdata/wfs_image.bin");
    array2device(wfs);
    ARRAY* subs = array_load_file("../testdata/wfs_subs.bin");
    array2device(subs);
    ARRAY* patch = array_load_file("../testdata/wfs_patch_index.bin");
    array2device(patch);

    ARRAY* nfft = array_load_file("../testdata/wfs_nbig.bin");
    int Ws = 10;
    int Hs = 10;
    int W = nfft->data[0];
    int H = nfft->data[0];
    int Wp = phase->shape[1];
    int nsub = subs->shape[0];

    ARRAY* fft_res = array_zeros(3, nsub, W, H);
    
    cufftComplex *d_Data;
    cudaMalloc((void**)&d_Data, W * nsub * H * sizeof(cufftComplex));

    cufftComplex *fd_Data;
    cudaMalloc((void**)&fd_Data, W * nsub * H * sizeof(cufftComplex));
    cufftComplex *f_data = (cufftComplex*)malloc(W * H * nsub * sizeof(cufftComplex));
    
    ARRAY* wfs_res = array_zeros(2, patch->shape[0], patch->shape[0]);
    array2device(wfs_res);

    int threadsPerBlock = 128;
    int blocksPerGrid =
            (H * nsub + threadsPerBlock - 1) / threadsPerBlock;

    SetFFTInput<<<blocksPerGrid, threadsPerBlock>>>(d_Data, phase->gdata, subs->gdata, H, W, Hs, Ws, Wp, nsub);
    cudaDeviceSynchronize();

    const int rank = 2;//维数
    int n[rank] = {H, W};//n*m
    int*inembed = n;//输入的数组sizecudaMemcpyHostToDevice
    int istride = 1;//数组内数据连续，为1
    int idist = n[0] * n[1];//1个数组的内存大小
    int*onembed = n;//输出是一个数组的size
    int ostride = 1;//每点DFT后数据连续则为1
    int odist = n[0] * n[1];//输出第一个数组与第二个数组的距离，即两个数组的首元素的距离
    int batch = nsub;//批量处理的批数
    
    cufftHandle fftplanfwd; //创建句柄
    checkCudaErrors(
        cufftPlanMany(&fftplanfwd, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, batch));//针对多信号同时进行FFT

    cudaEvent_t start, stop;
    cudaEventCreate(&start);    //创建Event
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);    //记录当前时间


        checkCudaErrors(
            cufftExecC2C(fftplanfwd, d_Data, fd_Data, CUFFT_FORWARD));  
        cudaDeviceSynchronize();
  

    FFTpatch<<<blocksPerGrid, threadsPerBlock>>>(wfs_res->gdata, fd_Data, patch->gdata, 
            patch->shape[0], patch->shape[2]);

    cudaDeviceSynchronize();
    array2host(wfs_res);
    
    cudaEventRecord(stop, 0); 
    cudaEventSynchronize(start);    //Waits for an event to complete.
    cudaEventSynchronize(stop);   
    cudaEventElapsedTime(&time_elapsed, start, stop);
    cudaDeviceSynchronize();
    printf("执行时间：%f(ms)\n", time_elapsed);

    checkCudaErrors(
        cudaMemcpy(f_data, fd_Data, H * W * nsub * sizeof(cufftComplex), cudaMemcpyDeviceToHost));


    for (int i = 0; i < fft_res->size; i++){
        cufftComplex point = f_data[i];
        fft_res->data[i] = point.x * point.x + point.y * point.y;
    }

    FILE *fp;
    if(!(fp = fopen("../testdata/output.bin", "wb"))){
        printf("array file error!");
    }
    array_save(fp, fft_res);
    fclose(fp);

    if(!(fp = fopen("../testdata/wfs_out.bin", "wb"))){
        printf("array file error!");
    }
    array_save(fp, wfs_res);
    fclose(fp);

    checkCudaErrors(cufftDestroy(fftplanfwd));
    array_del(phase);
    array_del(wfs);
    array_del(fft_res);
    array_del(wfs_res);
    free(f_data);

    cudaFree(d_Data);
    cudaFree(fd_Data);
}
