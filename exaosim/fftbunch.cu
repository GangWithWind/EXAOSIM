#include <cufft.h>
#include <iostream>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <stdio.h>
using namespace std;
#define CHANNEL_NUM  10 //通道数、FFT次数
const int dataH = 512; //图像高度
const int dataW = 512;  //图像宽度
cufftHandle fftplanfwd; //创建句柄


__global__ void SetFFTInput(cufftComplex* input, int H, int W, int Nb)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int ix = 0, ib = 0;
    if (i < H * Nb){
        ix = i % H;
        ib = i / H + 10;
        if(ix < ib){
            for(int j = 0; j < ib; j++){
                input[i * W + j].y = 1;
            }
        }

    }
}


int main(void){
 /* 开辟主机端的内存空间 */
 printf("文件名planmany_cuda31.cu...\n");
 printf("分配CPU内存空间...\n");
 cufftComplex *h_Data = (cufftComplex*)malloc(dataH*CHANNEL_NUM*dataW* sizeof(cufftComplex));//可用cudaMallocHost设置
 cufftComplex *h_resultFFT = (cufftComplex*)malloc(dataH*CHANNEL_NUM*dataW* sizeof(cufftComplex));
 /* 开辟设备端的内存空间 */
 printf("分配GPU内存空间...\n");
 /* 定义设备端的内存空间 */
 cufftComplex *d_Data;//device表示GPU内存，存储从cpu拷贝到GPU的数据
 cufftComplex *fd_Data;//device表示GPU内存,R2C后存入cufftComplex类型数据
 checkCudaErrors(cudaMalloc((void**)&d_Data, dataH*CHANNEL_NUM*dataW* sizeof(cufftComplex)));
 checkCudaErrors(cudaMemset(d_Data, 0, dataH*CHANNEL_NUM * dataW* sizeof(cufftComplex))); // 初始为0
 checkCudaErrors(cudaMalloc((void**)&fd_Data, dataH*CHANNEL_NUM*dataW* sizeof(cufftComplex))); // 开辟R2C后的设备内存
 checkCudaErrors(cudaMemset(fd_Data, 0, dataH*CHANNEL_NUM*dataW* sizeof(cufftComplex))); // 初始为0
 //随机初始化测试数据
 printf("初始化测试数据...\n");
 for (int k = 0; k < CHANNEL_NUM; k++){
    for (int i = 0; i < dataH; i++){
        for (int j = 0; j < dataW; j++){
            h_Data[(i + k * dataH) * dataW + j].x = 0;//float(rand()%255);
            h_Data[(i + k * dataH) * dataW + j].y = 0;//float(rand()%255);
            if(i < (4 + k) && j < (4 + k)){
                h_Data[(i + k * dataH)*dataW + j].x = 1;//float(rand()%255);
            }

        }
    }
}

int threadsPerBlock = 256;
int blocksPerGrid =
        (dataH * CHANNEL_NUM + threadsPerBlock - 1) / threadsPerBlock;
SetFFTInput<<<blocksPerGrid, threadsPerBlock>>>(d_Data, dataH, dataW, CHANNEL_NUM);
cudaDeviceSynchronize();

 //使用event计算时间
 float time_elapsed = 0;
 cudaEvent_t start, stop;
 cudaEventCreate(&start);    //创建Event
 cudaEventCreate(&stop);
 const int rank = 2;//维数
 int n[rank] = { dataH, dataW };//n*m
 int*inembed = n;//输入的数组sizecudaMemcpyHostToDevice
 int istride = 1;//数组内数据连续，为1
 int idist = n[0] * n[1];//1个数组的内存大小
 int*onembed = n;//输出是一个数组的size
 int ostride = 1;//每点DFT后数据连续则为1
 int odist = n[0] * n[1];//输出第一个数组与第二个数组的距离，即两个数组的首元素的距离
 int batch = CHANNEL_NUM;//批量处理的批数
 //采用cufftPlanMany方法
 checkCudaErrors(
  cufftPlanMany(&fftplanfwd, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, batch));//针对多信号同时进行FFT
 printf("拷贝CPU数据到GPU中...\n");
//  checkCudaErrors(
//   cudaMemcpy(d_Data, h_Data, dataW * dataH*CHANNEL_NUM * sizeof(cufftComplex), cudaMemcpyHostToDevice));
//  //printf("执行R2C-FFT...\n");
 printf("开始计时...\n");
 cudaEventRecord(start, 0);    //记录当前时间
 checkCudaErrors(
  cufftExecC2C(fftplanfwd, d_Data, fd_Data, CUFFT_FORWARD));
 cudaEventRecord(stop, 0);    //记录当前时间
 cudaEventSynchronize(start);    //Waits for an event to complete.
 cudaEventSynchronize(stop);    //Waits for an event to complete.Record之前的任务
 cudaEventElapsedTime(&time_eaAlapsed, start, stop);    //计算时间差
 cudaDeviceSynchronize();
 printf("拷贝GPU数据返回到CPU中...\n");
 checkCudaErrors(
  cudaMemcpy(h_resultFFT, fd_Data, dataW *dataH*CHANNEL_NUM * sizeof(cufftComplex), cudaMemcpyDeviceToHost));//将fft后的数据拷贝回主机
 printf("显示返回到CPU中的数据...\n");

FILE *fp;
fp = fopen("test.txt", "w");

for (int i = 0; i < dataH*CHANNEL_NUM*dataW; i++){
    fprintf(fp, "%.10f\n", h_resultFFT[i].x*h_resultFFT[i].x + h_resultFFT[i].y*h_resultFFT[i].y);
    //  cout << "h_resultFFT[" << i << "]=" << h_resultFFT[i].x << " + " << h_resultFFT[i].y << " i" << endl;
}
fclose(fp);

 cudaEventDestroy(start);    //destory the event
 cudaEventDestroy(stop);
 printf("执行时间：%f(ms)\n", time_elapsed);
 /* 销毁句柄 */
 checkCudaErrors(cufftDestroy(fftplanfwd));
 /* 释放设备空间 */
 checkCudaErrors(cudaFree(d_Data));
 checkCudaErrors(cudaFree(fd_Data));
 free(h_Data);
 free(h_resultFFT);
 return 0;
}