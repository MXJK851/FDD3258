#include <stdio.h>
#define N 256
#define TPB 256

__global__ void cuda_hello()
{
  const int myID = blockIdx.x*blockDim.x+threadIdx.x;
  printf("Hello World! My threadId is %d\n",myID);
}

int main()
{ 
  // Allocate device memory to store the output array
  // cudaMalloc(&d_out, N*sizeof(float));
  
  // Launch kernel to compute and store distance values
  cuda_hello<<<N/TPB, TPB>>>();
  
  cudaDeviceSynchronize();
  
  // cudaFree(d_out); // Free the memory
  
  return 0;
}
