#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include "../inc/g_array.h"

ARRAY* array_zeros(int s_num, ...){
    va_list valist;

    int num = abs(s_num);

    int *shape = (int *) malloc(num * sizeof(int));

    va_start(valist, s_num);

    int all = 1;
    if(s_num > 0){
        for(int i = 0; i < num; i++){
            shape[i] = va_arg(valist, int);
            all = all * shape[i];
        }
    }else{
        int *spp = va_arg(valist, int*);
        for(int i = 0; i < num; i++){
            shape[i] = spp[i];
            all = all * shape[i];
        }
    }

    ATYPE *data = (ATYPE *) malloc(all * sizeof(ATYPE));

    for(long i = 0; i < all; i++){
        data[i] = 0;
    }

    ARRAY *array = (ARRAY *) malloc(sizeof(ARRAY));
    array->dim = num;
    array->shape = shape;
    array->size = all;
    array->type = 'f';
    array->data = data;
    array->gpu_array = 0;
    array->gdata = NULL;
    return array;
}

ATYPE array_get(ARRAY* array, ...){
    va_list valist;
    va_start(valist, array);
    int num = array->dim;
    long index = 0;

    for(int i = 0; i < num; i++){
        index = index * array->shape[i] + va_arg(valist, int);
    }
    return array->data[index];
}

void array_set(ARRAY* array, ...){
    va_list valist;
    va_start(valist, array);
    int num = array->dim;
    long index = 0;

    for(int i = 0; i < num; i++){
        index = index * array->shape[i] + va_arg(valist, int);
    }
    array->data[index] = va_arg(valist, int);
}

ATYPE* array_ref(ARRAY* array, ...){
    va_list valist;
    va_start(valist, array);
    int num = array->dim;
    long index = 0;

    for(int i = 0; i < num; i++){
        index = index * array->shape[i] + va_arg(valist, int);
    }
    return &(array->data[index]);
}

ARRAY* array_load(FILE *file){
    int num;
    fread(&num, sizeof(int), 1, file);
    int shape[num];
    fread(shape, sizeof(int), num, file);

    ARRAY *array = array_zeros(-num, shape);
    fread(&(array->size), sizeof(int), 1, file);
    fread(&(array->type), sizeof(char), 1, file);
    fread(array->data, sizeof(ATYPE), array->size, file);

    return array;
}

void array_save(FILE *file, ARRAY *array){
    
    fwrite(&(array->dim), sizeof(int), 1, file);
    fwrite((array->shape), sizeof(int), array->dim, file);
    fwrite(&(array->size), sizeof(int), 1, file);
    fwrite(&(array->type), sizeof(char), 1, file);

    long all = 1;
    for(int i = 0; i < array->dim; i++){
        all = all * array->shape[i];
    }
    fwrite((array->data), sizeof(ATYPE), all, file);
}

void array_del(ARRAY *array){
    free(array->shape);
    free(array->data);
    free(array);

    if(array->gpu_array){
        cudaFree(array->gdata);
    }
}

ARRAY* array_zeros_like(ARRAY *array){
    return array_zeros(-(array->dim), array->shape);
}

ARRAY* array_load_file(char const* filename){
    FILE *fp;
    if(!(fp = fopen(filename, "rb"))){
        printf("array file '%s' error!", filename);
        exit(1);
    }
    ARRAY *array = array_load(fp);
    fclose(fp);
    return array;
}

void array2device(ARRAY *array){
    cudaMalloc((void**)&(array->gdata), array->size * sizeof(ATYPE));
    cudaMemcpy(array->gdata, array->data, array->size*sizeof(ATYPE), cudaMemcpyHostToDevice);
    array->gpu_array = 1;
}

void array2host(ARRAY *array){
    cudaMemcpy(array->data, array->gdata, array->size*sizeof(ATYPE), cudaMemcpyDeviceToHost);
}

__global__ void array_multi(float *A, float * B, float *C, int Ax, int Ay, 
int Bx, int By, int Cx, int Cy){
    if((Ay != Bx) || (Ax != Cx) || (By != Cy)){
        printf("size error!\nA shape: ,y: %d, x: %d\nB shape: y: %d, x: %d\nC shape: y: %d, x: %d\n",
        Ay, Ax, By, Bx, Cy, Cx);
    }
    int ib = blockDim.x * blockIdx.x + threadIdx.x;
    if(ib < Cx * Cy){
        int cy = ib / Cx;
        int cx = ib - cy * Cx;
        float sum;
        for(int i = 0; i < Ay; i++){
            sum += A[i * Ax + cx] * B[cy * Bx + i];
        }
        C[ib] = sum;
    }
}

void array_multiple_c(ARRAY *A, ARRAY *B, ARRAY *C){
// A with shape(ny, nx), B with shape(nz, ny), C with shape(nz, nx)

    if((A->shape[0] != B->shape[1]) || (A->shape[1] != C->shape[1]) || (B->shape[0] != C->shape[0])){
        printf("size error");
        printf("A shape: y: %d, x: %d\n", A->shape[0], A->shape[1]);
        printf("B shape: y: %d, x: %d\n", B->shape[0], B->shape[1]);
        printf("C shape: y: %d, x: %d\n", C->shape[0], C->shape[1]);
        exit(1);
    }

    float sum = 0;

    for(int z = 0; z < B->shape[0]; z++){
        for(int x = 0; x < A->shape[1]; x++){
            sum = 0;
            for(int y = 0; y < A->shape[0]; y++){
                sum += array_get(A, y, x) * array_get(B, z, y);
            }
            *array_ref(C, z, x) = sum;
        }
    }
}