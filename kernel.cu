
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

using namespace std;

__global__ void print() {
    printf("%d %d %d \n", threadIdx.x, threadIdx.y, threadIdx.z);
    printf("%d %d %d \n", blockIdx.x, blockIdx.y, blockIdx.z);
    printf("%d %d %d \n", gridDim.x, gridDim.y, gridDim.z);
}

int main()
{
    dim3 block(2, 2, 2);
    dim3 grid(4 / block.x, 4 / block.y, 4 / block.z);

    print << <grid, block >> > ();
    
    return 0;
}
