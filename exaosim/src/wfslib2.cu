#include <stdio.h>
#include <cufft.h>
#include <cuda_runtime.h>
#include "../inc/g_array.h"

extern "C" void cuwfs_plan(int n_big, int sub_sz, int nsub, int phase_sz, int wfs_sz, int patcah_sz, 
    int* subs, int* pupil, int* patch);
extern "C" void cuwfs_run(float* wfs_img, float* phase, int index);
extern "C" void cuwfs_destroy();

typedef struct{
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
    int blocksPerGrid_fftin;
    int blocksPerGrid_fftout;
    int index;

}  WFSPLAN;

int g_n_wfs = 0;
WFSPLAN **plans;


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

    g_n_wfs += 1;
    WFSPLAN** plans0 = plans;
    plans = (WFSPLAN**) malloc(g_n_wfs * sizeof(WFSPLAN*));

    for(int i = 0; i < g_n_wfs - 1; i++){
        plans[i] = plans0[i];
    }
    WFSPLAN* cp = (WFSPLAN*) malloc(sizeof(WFSPLAN));
    plans[g_n_wfs - 1] = cp;
    free(plans0);

    cp->index = g_n_wfs - 1;
    cp->g_n_big = n_big;
    cp->g_sub_sz = sub_sz;
    cp->g_nsub = nsub;
    cp->g_phase_sz = phase_sz;
    cp->g_wfs_sz = wfs_sz;
    cp->g_patch_sz = patch_sz;
    

    //initial device memory

    cudaMalloc((void**)&cp->d_subs, 2 * nsub * sizeof(int));
    cudaMemcpy(cp->d_subs, subs, 2 * nsub * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&cp->d_patch, wfs_sz * wfs_sz * patch_sz * sizeof(int));
    cudaMemcpy(cp->d_patch, patch, wfs_sz * wfs_sz * patch_sz * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&cp->d_pupil, sub_sz * sub_sz * nsub* sizeof(int));
    cudaMemcpy(cp->d_pupil, pupil, sub_sz * sub_sz * nsub* sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&cp->d_fft_input, n_big * n_big * nsub * sizeof(cufftComplex));
    cudaMemset(cp->d_fft_input, 0, n_big * n_big * nsub * sizeof(cufftComplex));

    cudaMalloc((void**)&cp->d_fft_output, n_big * n_big * nsub * sizeof(cufftComplex));
    cudaMalloc((void**)&cp->d_wfs, wfs_sz * wfs_sz * sizeof(float));
    cudaMalloc((void**)&cp->d_phase, phase_sz * phase_sz * sizeof(float));



    cp->threadsPerBlock = 128;
    cp->blocksPerGrid_fftin = (n_big * nsub + cp->threadsPerBlock - 1) / cp->threadsPerBlock;
    cp->blocksPerGrid_fftout = (wfs_sz * wfs_sz + cp->threadsPerBlock - 1) / cp->threadsPerBlock;
    
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


    cufftPlanMany(&cp->fftplanfwd, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, batch);
    
}

void share_phase(float *phase){
    if(g_n_wfs > 0){
        WFSPLAN *cp = plans[0];
        cudaMemcpy(cp->d_phase, phase, cp->g_phase_sz * cp->g_phase_sz * sizeof(float), cudaMemcpyHostToDevice);
    }
}

void cuwfs_run(float* wfs_img, float* phase, int index){

    if(index >= g_n_wfs){
        return;
    }


    WFSPLAN *cp = plans[0];
    cudaMemcpy(cp->d_phase, phase, cp->g_phase_sz * cp->g_phase_sz * sizeof(float), cudaMemcpyHostToDevice);
    SetFFTInput<<<cp->blocksPerGrid_fftin, cp->threadsPerBlock>>>(cp->d_fft_input, cp->d_phase, cp->d_subs, cp->d_pupil,
        cp->g_n_big, cp->g_n_big, cp->g_sub_sz, cp->g_sub_sz, cp->g_phase_sz, cp->g_nsub);
    cudaDeviceSynchronize();

    // for(int i = 0; i < 10; i++){
    //     printf("%f ", phase[i]);
    // }

    cufftExecC2C(cp->fftplanfwd, cp->d_fft_input, cp->d_fft_output, CUFFT_FORWARD);
    cudaDeviceSynchronize();
    
    FFTpatch<<<cp->blocksPerGrid_fftout, cp->threadsPerBlock>>>(cp->d_wfs, cp->d_fft_output, cp->d_patch, 
        cp->g_wfs_sz, cp->g_patch_sz);
    cudaDeviceSynchronize();

    cudaMemcpy(wfs_img, cp->d_wfs, cp->g_wfs_sz * cp->g_wfs_sz * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    //output fft for check
    // cufftComplex* fft_output = (cufftComplex*)malloc(cp->g_nsub * cp->g_n_big * cp->g_n_big * sizeof(cufftComplex));
    // ARRAY *fft_res = array_zeros(3, cp->g_nsub, cp->g_n_big, cp->g_n_big);
    // cudaMemcpy(fft_output, cp->d_fft_output, cp->g_nsub * cp->g_n_big * cp->g_n_big * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

    // for(int i = 0; i < fft_res->size; i++){
    //     fft_res->data[i] = fft_output[i].x * fft_output[i].x + fft_output[i].y * fft_output[i].y;
    // }

    // FILE *fp;
    // if(!(fp = fopen("middle.bin", "wb"))){
    //     printf("array file error!");
    // }
    // array_save(fp, fft_res);
    // fclose(fp);

}


void cuwfs_destroy(){
    for(int i = 0; i < g_n_wfs; i++){
        WFSPLAN *cp = plans[i];
        cudaFree(cp->d_subs);
        cudaFree(cp->d_patch);
        cudaFree(cp->d_fft_output);
        cudaFree(cp->d_fft_input);
        cudaFree(cp->d_wfs);
        cudaFree(cp->d_pupil);
        cudaFree(cp->d_phase);
        cufftDestroy(cp->fftplanfwd);
        free(cp);
    }
    free(plans);
}