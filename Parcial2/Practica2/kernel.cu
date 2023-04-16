#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

using namespace std;

void bubbleSort(int* arr, int size) {
    for (int i = 0; i < size - 1; i++) {
        for (int j = 0; j < size - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                int aux = arr[j + 1];
                arr[j + 1] = arr[j];
                arr[j] = aux;
            }
        }
    }
}

__global__ void bubbleSortGPU(int* arr, int size) {
    int tid = threadIdx.x;

    for (int i = 0; i < size; i++) {
        int offset = i % 2;
        if (2 * tid + offset + 1 < size) {
          if (arr[2 * tid + offset] > arr[2 * tid + offset + 1]) {
            int aux = arr[2 * tid + offset];
            arr[2 * tid + offset] = arr[2 * tid + offset + 1];  
            arr[2 * tid + offset + 1] = aux;
          }
        }
        __syncthreads();
    }
}

int main() {
    int size = 15;
    int* host_a, * ans, * dev_a;
    host_a = (int*)malloc(size * sizeof(int));
    ans = (int*)malloc(size * sizeof(int));
    cudaMalloc(&dev_a, size * sizeof(size));

    for (int i = 0; i < size; i++) {
        host_a[i] = (rand() % (256));
        printf("%d ", host_a[i]);
    }
    printf("\n");

    cudaMemcpy(dev_a, host_a, size * sizeof(int), cudaMemcpyHostToDevice);

    dim3 grid(1);
    dim3 block(size);
    bubbleSortGPU << <grid, block >> > (dev_a, size);
    cudaMemcpy(ans, dev_a, size * sizeof(int), cudaMemcpyDeviceToHost);

    bubbleSort(host_a, size);

    printf("CPU: ");
    for (int i = 0; i < size; i++) {
        printf("%d ", host_a[i]);
    }
    printf("\n");
    printf("GPU: ");
    for (int i = 0; i < size; i++) {
        printf("%d ", ans[i]);
    }
    printf("\n");
    return 0;
}
