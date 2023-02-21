#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

__global__ void idx_calc_gid_3D(int* input) {
    int totalThreads = blockDim.x * blockDim.y * blockDim.z;
    int tid = threadIdx.x // 1D
        + threadIdx.y * blockDim.x // 2D
        + threadIdx.z * blockDim.x * blockDim.y; // 3D
    int bid = blockIdx.x // 1D
        + blockIdx.y * gridDim.x // 2D
        + blockIdx.z * gridDim.x * gridDim.y; // 3D
    int gid = tid + bid * totalThreads;
    printf("[DEVICE] gid: %d, data: %d\n\r", gid, input[gid]);
}

__global__ void sum_array_gpu(int* a, int* b, int* c, int size) {
    int totalThreads = blockDim.x * blockDim.y * blockDim.z;
    int tid = threadIdx.x // 1D
        + threadIdx.y * blockDim.x // 2D
        + threadIdx.z * blockDim.x * blockDim.y; // 3D
    int bid = blockIdx.x // 1D
        + blockIdx.y * gridDim.x // 2D
        + blockIdx.z * gridDim.x * gridDim.y; // 3D
    int gid = tid + bid * totalThreads;
    if (gid < size) {
        c[gid] = a[gid] + b[gid];
    }
}

void sum_array_cpu(int* a, int* b, int* c, int size) {
    for (int i = 0; i < size; i++)
    {
        c[i] = a[i] + b[i];
    }
}

__global__ void sum_array_gpu_3(int* a, int* b, int* c, int* ans, int size) {
    int totalThreads = blockDim.x * blockDim.y * blockDim.z;
    int tid = threadIdx.x // 1D
        + threadIdx.y * blockDim.x // 2D
        + threadIdx.z * blockDim.x * blockDim.y; // 3D
    int bid = blockIdx.x // 1D
        + blockIdx.y * gridDim.x // 2D
        + blockIdx.z * gridDim.x * gridDim.y; // 3D
    int gid = tid + bid * totalThreads;
    if (gid < size) {
        ans[gid] = a[gid] + b[gid] + c[gid];
    }
}

void sum_array_cpu_3(int* a, int* b, int* c, int* ans, int size) {
    for (int i = 0; i < size; i++)
    {
        ans[i] = a[i] + b[i] + c[i];
    }
}

int main()
{
    const int N = 10000;

    int a[N];
    int b[N];
    int c[N];
    int sumGPU[N];
    int sumCPU[N];
    bool meow = true;

    for (int i = 0; i < N; i++) {
        a[i] = rand() % 256;
        b[i] = rand() % 256;
        c[i] = rand() % 256;
    }

    int size = N * sizeof(int);

    int* d_a = 0;
    int* d_b = 0;
    int* d_c = 0;
    int* d_sumGPU;

    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_sumGPU, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, size, cudaMemcpyHostToDevice);
  
    clock_t gpu_start, gpu_stop;  
  
    gpu_start = clock();
    sum_array_gpu_3 << <79, 128 >> > (d_a, d_b, d_c d_sumGPU, N);
    sum_array_cpu_3(a, b, c, sumCPU, N);
    cudaDeviceSynchronize();
    gpu_stop = clock();
    cudaDeviceReset();

    double cps_gpu = (double)((double)(gpu_stop - gpu_start) / CLOCKS_PER_SEC);
    printf("Execution Time [ET-GPU]: %4.6f\n\r", cps_gpu);

    cudaMemcpy(sumGPU, d_sumGPU, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        if (sumGPU[i] != sumCPU[i]) {
            meow = false;
            break;
        }
    }

    if (meow) {
        printf("Both arrays are equal");
    }
    else {
        printf("The arrays are not equal");
    }

    cudaDeviceReset();

    cudaFree(d_a);
    return 0;
}
