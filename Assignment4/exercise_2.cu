#include <stdio.h>
#include <math.h>
#define ARRAY_SIZE 97
#define TPB 32

__host__ float cpu_saxpy(int i, float a, float *X, float *Y)
{
  return (a*X[i]+Y[i]);
}

__device__ float gpu_saxpy(int i, float a, float *X, float *Y)
{
  return (a*X[i]+Y[i]);
}

__global__ void ThreadId(float *y_out, int n, float a, float *X, float *Y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  if (i<n)
  {
    y_out[i]=gpu_saxpy(i,a,X,Y);
  } 

}

int main()
{
  float a=2.0,X[ARRAY_SIZE],*x=0,Y[ARRAY_SIZE],*y=0,Y_out[ARRAY_SIZE],*y_out=0;
 
  int j;
  
  for (j=0;j<ARRAY_SIZE;j++)
  {
    X[j]=j;
    Y[j]=j;
    Y_out[j]=cpu_saxpy(j,a,X,Y);
    // printf("%f\n",Y_out[j]);
  }

  cudaMalloc(&y_out, ARRAY_SIZE*sizeof(float));
  cudaMalloc(&x, ARRAY_SIZE*sizeof(float));
  cudaMalloc(&y, ARRAY_SIZE*sizeof(float));

  cudaMemcpy(x, X, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(y, Y, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice);
  
  int num_block;
  num_block= ARRAY_SIZE/TPB;
  while (ARRAY_SIZE>num_block*TPB)
  {
    num_block++;
  }

  //num_block=ceil(float(ARRAY_SIZE/TPB));
  //printf("%d\n",num_block);
  //ThreadId<<<ceil(ARRAY_SIZE/TPB), TPB>>>(y_out, ARRAY_SIZE, a, x, y);
  ThreadId<<<num_block,TPB>>>(y_out, ARRAY_SIZE, a, x, y);

  cudaDeviceSynchronize();  
 
  //float *y=(float*)malloc(ARRAY_SIZE*sizeof(float));
  float z[ARRAY_SIZE];
  
  for (j=0;j<ARRAY_SIZE;j++)
  {
    cudaMemcpy(z, y_out, ARRAY_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
  }

  cudaFree(y_out);
  cudaFree(x);
  cudaFree(y);
 
  for (j=0;j<ARRAY_SIZE;j++)
  { 
    printf("CPU:%4.1f  GPU:%4.1f  Compare:%4.1f\n",Y_out[j],z[j],Y_out[j]-z[j]);
  }

  return 0;
}
