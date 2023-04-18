#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>

using namespace std;

__global__ void search(int* l, int* s, int* id) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < 32) {    
      if (l[tid] == s[0]) {
          *id = tid;
      }
    }
}

int main() {
    int size = 32;
    int* host_a, * host_s, * host_id;
    int* dev_a, *dev_s, *dev_id;
    host_a = (int*)malloc(size * sizeof(int));
    host_s = (int*)malloc(size * sizeof(int));
    host_id = (int*)malloc(size * sizeof(int));
    host_s[0] = 4;
    host_id[0] = -1;
    cudaMalloc(&dev_a, size * sizeof(int));
    cudaMalloc(&dev_s, sizeof(int));
    cudaMalloc(&dev_id, sizeof(int));

    for (int i = 0; i < size; i++) {
        int r1 = (rand() % (5));
        host_a[i] = r1;
        printf("%d ", host_a[i]);
    }
    printf("\n");
    
    cudaMemcpy(dev_a, host_a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_s, host_s, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_id, host_id, sizeof(int), cudaMemcpyHostToDevice);
    
    dim3 grid(size>=1024 ? size/1024:1);
    dim3 block(1024);
    search << <grid, block >> > (dev_a, dev_s, dev_id);
    cudaDeviceSynchronize();
    
    cudaMemcpy(host_id, dev_id, sizeof(int), cudaMemcpyDeviceToHost);

    if (host_id[0] == -1) {
        printf("Not Found\n");
    }
    else {
        printf("Found at %d index\n", host_id[0]);
    }
    return 0;
}
