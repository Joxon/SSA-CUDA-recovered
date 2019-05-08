#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>    
#include <device_launch_parameters.h>

// 0+1+2+...+SIZE
#define SIZE 10

__global__ void histo_kernel(int size, unsigned int *histo)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < size)
    {
         //*histo += i;  
        atomicAdd(histo, i);
    }
}

int main(void)
{
    int sum = 0;

    //�����ڴ沢������ʼ����  
    unsigned int *dev_histo;
    cudaMalloc((void**)&dev_histo, sizeof(int));
    cudaMemcpy(dev_histo, &sum, sizeof(int), cudaMemcpyHostToDevice);

    // kernel launch - 2x the number of mps gave best timing    
    cudaDeviceProp  prop;
    cudaGetDeviceProperties(&prop, 0);
    int blocks = prop.multiProcessorCount;

    //ȷ���߳����㹻  
    histo_kernel <<<blocks * 2, (SIZE + 2 * blocks - 1) / blocks / 2 >>> (SIZE, dev_histo);

    //���ݿ�����CPU�ڴ�  
    cudaMemcpy(&sum, dev_histo, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Threads sum = %d\n", sum);

    cudaFree(dev_histo);
    return 0;
}