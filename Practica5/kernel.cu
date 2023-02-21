
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

using namespace std;

#define GPUErrorAssertion(ans) {gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n\r", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void multMat(int* a, int* b, int* c, int width, int rows, int cols) { 
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int aux = 0;

    if (row < rows && col < cols) {
        for (int i = 0; i < width; i++) {
            aux += a[row * width + i] * b[i * width + col];
        }
        c[row * width + col] = aux;
    }
}

int main()
{
    int rows = 64;
    int cols = 32;
    int Arows = rows;
    int Acols = cols;
    int Brows = cols;
    int Bcols = rows;
    int Crows = rows;
    int Ccols = rows;

    int Abytes = Arows * Acols * sizeof(int);
    int Bbytes = Brows * Bcols * sizeof(int);
    int Cbytes = Crows * Ccols * sizeof(int);
    int blockSize = 2;

    int Csize = Crows * Ccols;

    int* h_a, * h_b, * h_c, * gpu_res;

    h_a = (int*)malloc(Abytes);
    h_b = (int*)malloc(Bbytes);
    h_c = (int*)malloc(Cbytes);
    gpu_res = (int*)malloc(Cbytes);
    memset(gpu_res, 0, Cbytes);

    time_t t;
    srand((unsigned)time(&t));

    for (int i = 0; i < Arows; i++)
    {
        for (int j = 0; j < Acols; j++)
        {
            h_a[i * Acols + j] = rand() % 2;
            h_b[i * Bcols + j] = rand() % 2;
        }
    }

    int* d_a, * d_b, * d_c, * d_out;

    GPUErrorAssertion(cudaMalloc((int**)&d_a, Abytes));
    GPUErrorAssertion(cudaMalloc((int**)&d_b, Bbytes));
    GPUErrorAssertion(cudaMalloc((int**)&d_c, Cbytes));
    GPUErrorAssertion(cudaMalloc((int**)&d_out, Cbytes));

    GPUErrorAssertion(cudaMemcpy(d_a, h_a, Abytes, cudaMemcpyHostToDevice));
    GPUErrorAssertion(cudaMemcpy(d_b, h_b, Bbytes, cudaMemcpyHostToDevice));
    GPUErrorAssertion(cudaMemcpy(d_c, h_c, Cbytes, cudaMemcpyHostToDevice));
    GPUErrorAssertion(cudaMemcpy(d_out, gpu_res, Cbytes, cudaMemcpyHostToDevice));

    dim3 block(blockSize, blockSize);
    dim3 grid(ceil(Csize / blockSize), ceil(Csize / blockSize));

    clock_t gpu_start, gpu_stop;

    gpu_start = clock();
    multMat << <grid, block >> > (d_a, d_b, d_c, Acols, Crows, Ccols);
    cudaDeviceSynchronize();
    gpu_stop = clock();
    double cps_gpu = (double)((double)(gpu_stop - gpu_start) / CLOCKS_PER_SEC);
    printf("Execution time [ET_GPU]: %4.6f \n\r", cps_gpu);

    cudaMemcpy(d_c, h_c, Csize, cudaMemcpyDeviceToHost);

    for (int i = 0; i < Crows; i++)
    {
        for (int j = 0; j < Ccols; j++)
        {
            cout << h_c[i * blockSize + j] << " ";
        }
    }

    cudaDeviceReset();
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_out);
}
