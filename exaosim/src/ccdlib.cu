#include <stdio.h>
#include <cufft.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "../inc/g_array.h"

typedef struct{
    int n_big;
    int sub_sz;
    int phase_sz;
    int image_sz;
    int subx;
    int suby;

    float* d_phase;
    int* d_pupil;
    float* d_image;

    cufftComplex* d_fft_input;
    cufftComplex* d_fft_output;
    cufftHandle fftplanfwd;
    
}  CCDPLAN;

int g_n_ccd = 0;
CCDPLAN** ccds;

void ccd_plan(int n_big, int sub_sz, int phase_sz, int image_sz, int subx, int suby, 
int* pupil){
    g_n_ccd += 1;
    CCDPLAN** ccds0 = ccds;

    ccds = (CCDPLAN**) malloc(sizeof(CCDPLAN*));
    for(int i = 0; i < g_n_ccd - 1; i++){
        ccds[i] = ccds0[i];
    }
    CCDPLAN* cp = (CCDPLAN *) malloc(sizeof(CCDPLAN));
    ccds[g_n_ccd - 1] = cp;
    
    cp->n_big = n_big;
    cp->sub_sz = sub_sz;
    cp->phase_sz = phase_sz;
    cp->image_sz = image_sz;
    cp->subx = subx;
    cp->suby = suby;

    checkCudaErrors(
        cudaMalloc((void**)&cp->d_phase, phase_sz * phase_sz * sizeof(float)));
    
    cudaMalloc((void**)&cp->d_fft_input, n_big * n_big * sizeof(cufftComplex));
    cudaMemset(cp->d_fft_input, 0, n_big * n_big * sizeof(cufftComplex));
    cudaMalloc((void**)&cp->d_fft_output, n_big * n_big * sizeof(cufftComplex));
    cudaMalloc((void**)&cp->d_pupil, sub_sz * sub_sz * sizeof(int));
    cudaMalloc((void**)&cp->d_image, image_sz * image_sz * sizeof(int));
    cudaMemcpy(cp->d_pupil, pupil, sub_sz * sub_sz * sizeof(int), cudaMemcpyHostToDevice);
    cufftPlan2d(&(cp->fftplanfwd), n_big, n_big, CUFFT_C2C);
}


__global__ void setFFTInput(cufftComplex* fft_out, float* phase, int* pupil, int subx, int suby, 
int n_big, int sub_sz, int phase_sz){
    int i = blockDim.x *blockIdx.x + threadIdx.x;
    if(i < sub_sz * sub_sz){
        int ix = i / sub_sz;
        int iy = i % sub_sz;
        int ip = ix * phase_sz + iy;
        fft_out[ix * n_big + iy].x = cos(phase[ip]) * pupil[i];
        fft_out[ix * n_big + iy].y = sin(phase[ip]) * pupil[i];
    }
}

__global__ void getFFTimg(float* img, cufftComplex* fft_output, 
int image_sz, int n_big){

    int i = blockDim.x *blockIdx.x + threadIdx.x;
    int fx;
    int fy;
    if(i < image_sz * image_sz){
        int ix = i / image_sz;
        int iy = i % image_sz;
        
        if(ix >= image_sz / 2){
            fx = ix - image_sz/2;
        }else{
            fx = n_big + ix - image_sz/2;
        }

        if(iy >= image_sz / 2){
            fy = iy - image_sz / 2;
        }else{
            fy = n_big + iy - image_sz / 2;
        }

        img[i] = fft_output[fy + fx * n_big].x * fft_output[fy + fx * n_big].x;
        img[i] += fft_output[fy + fx * n_big].y * fft_output[fy + fx * n_big].y;
    }

}

void ccd_run_single(float* image, float* phase, int ccd_id){
    CCDPLAN* cp = ccds[ccd_id];
    checkCudaErrors(
    cudaMemcpy(cp->d_phase, phase, cp->phase_sz * cp->phase_sz * sizeof(float), cudaMemcpyHostToDevice));
    int threadsPerBlock = 128;
    int blocksPerGrid = (cp->sub_sz * cp->sub_sz + threadsPerBlock - 1) / threadsPerBlock;

    setFFTInput<<<blocksPerGrid, threadsPerBlock>>>(cp->d_fft_input, cp->d_phase, cp->d_pupil, 
    cp->subx, cp->suby,  cp->n_big, cp->sub_sz, cp->phase_sz);
    cudaDeviceSynchronize();

    checkCudaErrors(
    cufftExecC2C(cp->fftplanfwd, cp->d_fft_input, cp->d_fft_output, CUFFT_FORWARD));
    
    blocksPerGrid = (cp->image_sz * cp->image_sz + threadsPerBlock - 1) / threadsPerBlock;

    getFFTimg<<<blocksPerGrid, threadsPerBlock>>>(cp->d_image, cp->d_fft_output, cp->image_sz, cp->n_big);
    cudaDeviceSynchronize();

    // cufftComplex* fft_input = (cufftComplex*) malloc(cp->n_big * cp->n_big * sizeof(cufftComplex));
    // cudaMemcpy(fft_input, cp->d_fft_output, cp->n_big * cp->n_big * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

    // ARRAY* fftres = array_zeros(2, cp->n_big, cp->n_big);
    // for(int i = 0; i < fftres->size; i++){
    //     fftres->data[i] = fft_input[i].x * fft_input[i].x + fft_input[i].y * fft_input[i].y;
    // }



    checkCudaErrors(
    cudaMemcpy(image, cp->d_image, cp->image_sz * cp->image_sz * sizeof(float), cudaMemcpyDeviceToHost));
    
    cudaDeviceSynchronize();
}

int main(){
    int sub_sz = 30;
    int phase_sz = 40;
    int pupil[sub_sz * sub_sz];
    int n_big = 400;

    for(int i = 0; i < sub_sz * sub_sz; i++){
        pupil[i] = 1;
    }

    int image_sz = 51;
    int subx = 5;
    int suby = 5;

    
    ccd_plan(n_big, sub_sz, phase_sz, image_sz, subx, suby, pupil);
    ccd_plan(n_big*2, sub_sz, phase_sz, image_sz, subx, suby, pupil);

    float phase[phase_sz * phase_sz];
    for(int i = 0; i < phase_sz * phase_sz; i++){
        phase[i] = 0;
    }

    ARRAY * image = array_zeros(2, image_sz, image_sz);
    ccd_run_single(image->data, phase, 0);
    ccd_run_single(image->data, phase, 1);

    FILE *fp;
    if(!(fp = fopen("output.bin", "wb"))){
        printf("array file error!");
    }
    array_save(fp, image);
    fclose(fp);

}