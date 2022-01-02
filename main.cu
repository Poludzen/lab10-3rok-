
#include <iostream>
#include "cuda.h"




using namespace std;


static const int BLOCK_SIZE = 1024;

__inline__ __device__ double atomicAddD(double *address, double val) {
    auto* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    if (val==0.0)
        return __longlong_as_double(old);
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val +__longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}


__host__ __device__ double getMandelbrotPoint(int ind, int width, int height, double x0, double x1, double y0, double y1, int iter){
    int j = ind % width;
    int i = ind / width;
    long double a = i * (x1 - x0) / width + x0;
    long double b = j * (y1 - y0) / height + y0;
    long double ai = a;
    long double bi = b;
    int n = 0;
    for (int i1 = 0; i1 < iter; ++i1) {
        long double a1 = a * a - b * b;
        long double b1 = a * b * 2;

        a = a1 + ai;
        b = b1 + bi;

        if ((a + b) > 2) {
            break;
        }

        n++;
    }
}

__global__ void reduce1 (double * outData, int width, int height, double x0, double x1, double y0, double y1, int iter) {
    __shared__ double data [BLOCK_SIZE];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    data [tid] = getMandelbrotPoint(i, width, height, x0, x1, y0, y1, iter);
    for ( int s = 1; s < blockDim.x; s *= 2 )
    {
        if ( tid % (2 * s) == 0 ) data [tid] += data [tid + s];
        __syncthreads ();
    }
    if (tid == 0 )
        outData [blockIdx.x] = data [0];
}
__global__ void reduce2(double* outData, int width, int height, double x0, double x1, double y0, double y1, int iter){
    __shared__ double data [BLOCK_SIZE];
    data[threadIdx.x] = getMandelbrotPoint(blockIdx.x * blockDim.x + threadIdx.x, width, height, x0, x1, y0, y1, iter);
    for (int s = 1; s < BLOCK_SIZE; s*=2){
        int A1 = 2 * s * threadIdx.x;
        int A2 = A1 + s;
        if(A1 < BLOCK_SIZE){
            data[A1] += data[A2];
        }
        __syncthreads();
    }
    outData[blockIdx.x] = data[0];

}
__global__ void reduce3 (double * outData, int width, int height, double x0, double x1, double y0, double y1, int iter)
{
    __shared__ double data [BLOCK_SIZE];
    int tid = threadIdx.x;
    int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    data[tid] = getMandelbrotPoint(i, width, height, x0, x1, y0, y1, iter);
    __syncthreads ();
    for (int s = BLOCK_SIZE / 2; s > 0; s -= 1)
    {
        if ( tid < s ) {
            data[tid] += data[tid + s];
            data[tid + s] = 0;
        }
        __syncthreads ();
    }
    if ( tid == 0 ) {
        outData[blockIdx.x] = data[0];
        if(BLOCK_SIZE % 2 != 0)
           outData[blockIdx.x] += data[BLOCK_SIZE - 1];
    }
}
__global__ void reduce4 (double * outData, int width, int height, double x0, double x1, double y0, double y1, int iter)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    atomicAddD(&outData[blockIdx.x], getMandelbrotPoint(i, width, height, x0, x1, y0, y1, iter));
}
__global__ void reduce5(double* outData, int k, int width, int height, double x0, double x1, double y0, double y1, int iter){
    __shared__ double data[BLOCK_SIZE];
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    data[threadIdx.x] = getMandelbrotPoint(ind, width, height, x0, x1, y0, y1, iter);

    __syncthreads();

    if(threadIdx.x % k == 0) {
        for (int i = threadIdx.x + 1; i <= threadIdx.x + k - 1 ; i++) {
            atomicAddD(&data[threadIdx.x], data[i]);
        }
        __syncthreads();
        atomicAddD(&outData[blockIdx.x], data[threadIdx.x]);
    }
}
__global__ void reduce6 (double * outData, int width, int height, double x0, double x1, double y0, double y1, int iter) {
    __shared__ double data [BLOCK_SIZE];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    data [tid] = getMandelbrotPoint(i, width, height, x0, x1, y0, y1, iter);
    #pragma unroll 1024
    for ( int s = 1; s < blockDim.x; s *= 2 )
    {
        if ( tid % (2 * s) == 0 ) data [tid] += data [tid + s];
        __syncthreads ();
    }
    if (tid == 0 )
        outData [blockIdx.x] = data [0];
}
__global__ void reduce7(double* outData, int width, int height, double x0, double x1, double y0, double y1, int iter){
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    double val = getMandelbrotPoint(ind, width, height, x0, x1, y0, y1, iter);
    for ( int offset = BLOCK_SIZE / 2; offset > 0; offset /= 2 )
        val += __shfl_down_sync ( BLOCK_SIZE - 1, val, offset );

    if(threadIdx.x == 0){
        outData[0] = val;
    }
}


int main(){
    double arr1 = 0;
    int width = 1024, height = 1024, size = width * height / BLOCK_SIZE, iter = 64;
    double x0 = -0.82, x1 = 0.1, y0 = -0.7, y1 = 0.22;

    for (int i = 0; i < 10; i++) {
        clock_t start = clock();
        double *arr;
        cudaMalloc(&arr, sizeof(double) * size);
        reduce1<<<size, BLOCK_SIZE>>>(arr,width, height, x0,x1,y0,y1,iter);

        clock_t end = clock();
        arr1 += (double) (end - start) / CLOCKS_PER_SEC;
        auto *a = new double[size];
        cudaMemcpy(a, arr, sizeof(double) * size, cudaMemcpyDeviceToHost);
    }

    std::cout << arr1 / 10<< " ";
}