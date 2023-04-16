#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

using namespace std;

struct PointsAOS {
    int x;
    int y;
};

struct PointsSOA {
    int x[16];
    int y[16];
};

__global__ void AOS(PointsAOS* points, PointsAOS* results, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N) {
        PointsAOS aux = points[tid];
        aux.x = aux.x + 10;
        aux.y = aux.y + 10;
        results[tid] = aux;
    }
}

__global__ void SOA(PointsSOA* points, PointsSOA* results, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        results->x[tid] = points->x[tid] + 10;
        results->y[tid] = points->y[tid] + 10;
    }
}

int main() {
    /*
    // AOS
    int N = 16;
    int blockSize = 32;

    PointsAOS* h_points, * h_res;

    h_points = (PointsAOS*)malloc(sizeof(PointsAOS) * N);
    h_res = (PointsAOS*)malloc(sizeof(PointsAOS) * N);

    for (int k = 0; k < N; k++) {
        h_points[k].x = k + 1;
        h_points[k].y = k + 2;
    }

    PointsAOS* d_points, * d_results;
    cudaMalloc(&d_points, sizeof(PointsAOS) * N);
    cudaMalloc(&d_results, sizeof(PointsAOS) * N);

    cudaMemcpy(d_points, h_points, sizeof(PointsAOS) * N, cudaMemcpyHostToDevice);
    dim3 block(blockSize);
    dim3 grid((N+blockSize-1) / (block.x));
    AOS <<<grid, block>>> (d_points, d_results, N);

    cudaMemcpy(h_res, d_results, sizeof(PointsAOS) * N, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        printf("x: %d y: %d\n", h_res[i].x, h_res[i].y);
    }
    */

    //SOA
    int N = 16;
    int blockSize = 32;

    PointsSOA* h_points, * h_res;

    h_points = (PointsSOA*)malloc(sizeof(PointsSOA));
    h_res = (PointsSOA*)malloc(sizeof(PointsSOA));

    for (int k = 0; k < N; k++) {
        h_points->x[k] = k + 1;
        h_points->y[k] = k + 2;
    }

    printf("OG: \n");
    for (int i = 0; i < N; i++) {
        printf("x: %d y: %d\n", h_points->x[i], h_points->y[i]);
    }
    printf("\n");

    PointsSOA* d_points, * d_results;
    cudaMalloc(&d_points, sizeof(PointsSOA));
    cudaMalloc(&d_results, sizeof(PointsSOA));

    cudaMemcpy(d_points, h_points, sizeof(PointsSOA), cudaMemcpyHostToDevice);
    dim3 block(blockSize);
    dim3 grid((N + blockSize - 1) / (block.x));
    SOA << <grid, block >> > (d_points, d_results, N);

    cudaMemcpy(h_res, d_results, sizeof(PointsSOA), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        printf("x: %d y: %d\n", h_res->x[i], h_res->y[i]);
    }
}
