#include <stdio.h>

#define N 2048 * 2048// Number of elements in each vector

/*
 * Optimize this already-accelerated codebase. Work iteratively,
 * and use nsys to support your work.
 *
 * Aim to profile `saxpy` (without modifying `N`) running under
 * 200us.
 *
 * Some bugs have been placed in this codebase for your edification.
 */

__global__ void saxpy(int * a, int * b, int * c)
{
    int tid = blockIdx.x * blockDim.x  + threadIdx.x;

    if ( tid < N )
        c[tid] = 2 * a[tid] + b[tid];
}

__global__ void initWith(int * a, int val)
{
    int tid = blockIdx.x * blockDim.x  + threadIdx.x;

    if ( tid < N )
        a[tid] = val;
}


void checkElementsAre(int target, int* vector, int n)
{
  for(int i = 0; i < n; i++)
  {
    if(vector[i] != target)
    {
      printf("FAIL: vector[%d] - %d does not equal %d\n", i, vector[i], target);
      exit(1);
    }
  }
  printf("Success! All values calculated correctly.\n");
}

int main()
{
    int *a, *b, *c;

    size_t size = N * sizeof (int); // The total number of bytes per vector

    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);
    cudaMallocManaged(&c, size);

    cudaError_t asyncErr;

    int threads_per_block = 128;
    int number_of_blocks = (N / threads_per_block) + 1;

    int deviceId;
    cudaGetDevice(&deviceId);       

    cudaMemPrefetchAsync(a, size, deviceId);  
    cudaMemPrefetchAsync(b, size, deviceId);
    cudaMemPrefetchAsync(c, size, deviceId);

    initWith <<< number_of_blocks, threads_per_block >>> ( a, 2 );
    initWith <<< number_of_blocks, threads_per_block >>> ( b, 1 );
    initWith <<< number_of_blocks, threads_per_block >>> ( c, 0 );
    


    saxpy <<< number_of_blocks, threads_per_block >>> ( a, b, c );
    
    asyncErr = cudaDeviceSynchronize();
    if(asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));


    cudaMemPrefetchAsync(c, size, cudaCpuDeviceId);
    checkElementsAre(5, c, N);

    cudaFree( a ); cudaFree( b ); cudaFree( c );
}
