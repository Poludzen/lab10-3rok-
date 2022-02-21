%%cu
#include<stdio.h>
#include<time.h>
#include <iostream>

using namespace std;

__device__ float function1(float x) { return x*x; }
float f1(float x) { return x*x; }

__global__ void gpu_prost(float* a, float h, int N, float xp)
{ 
    int idx = blockIdx.x+1;
    if(idx <= N)
    {
        a[idx-1] = function1(xp + idx*h)*h;
    }
}

__global__ void gpu_trapezy(float* a, float h, int N, float xp)
{ 
    int idx = blockIdx.x+1;
    if(idx < N)
    {
        a[idx-1] = function1(xp + idx*h);
    }
}

__global__ void gpu_simpson(float* a, float* st, float h, int N, float xp)
{
    int idx = blockIdx.x+1;
    float x;
    if(idx <= N)
    {
      x = xp + idx * h;
      st[idx-1] = function1(x - h/2);
      if(idx < N) a[idx-1] = function1(x);
    }
}

float cudaProst(int N)
{
    cudaError_t status;
    size_t size = N * sizeof(float);
 
    float* a_h = (float*)malloc(size);
    float* a_h_sim = (float*)malloc(size);
    float* a_d;
    float *a_d_sim;

    float sum = 0;
    cudaMalloc((void**)&a_d, size);
    cudaMalloc((void**)&a_d_sim, size);
 
    double xp, xk;
 
    xp = -4;
    xk = 2;
    float h = (xk - xp) / (double)N;
    clock_t start, end;
    for(int vers = 3; vers < 4; vers++){
        start = clock();
        if(vers == 1){
            gpu_prost<<<N, 1>>>(a_d, h, N, xp);
        } else if(vers == 2){
            gpu_trapezy<<<N, 1>>>(a_d, h, N, xp);
        } else if(vers == 3){
            gpu_simpson<<<N, 1>>>(a_d, a_d_sim, h, N, xp);
        }

        // dla simpsona a_h - wartosc funkcji
        cudaMemcpy(a_h, a_d, sizeof(float)*N, cudaMemcpyDeviceToHost);
        // dla simpsona a_h_sim - step
        cudaMemcpy(a_h_sim, a_d_sim, sizeof(float)*N, cudaMemcpyDeviceToHost);

        if(vers == 1){
            for(int i = 0; i < N; i++){
                sum += a_h[i];
            }

        } else if(vers == 2){
            sum += f1(xp)/2;
            sum += f1(xk)/2;
            for(int i = 0; i <= N; i++){
                sum += a_h[i];
            }
            sum *= h;

        } else if(vers == 3){
            float calka=0, st=0;
            for(int i = 0; i <= N; i++){
                calka += a_h[i];
                //printf("calka: %f\n", calka);
                st += a_h_sim[i];
                //printf("St: %f\n", st);
            }
            sum = h / 6*(f1(xp) + f1(xk) + calka*2 + st*4);
        }
    
        end = clock();
       
        double gpu = ((double)(end-start))/CLOCKS_PER_SEC; 
        printf("GPU v-%d: %lf sekund\n", vers, gpu);
        free(a_h);
        cudaFree(a_d);
        free(a_h_sim);
        cudaFree(a_d_sim);
    }
    return sum;    
}

int main(){
    float sum=0;
    for(int k = 10; k <= 100000; k *= 10){
        printf("Wynik dla %d\n", k);
        for(int iter = 1; iter <= 20; iter++){
            sum = cudaProst(k);
            //printf("Wynik dla %d: %f\n", k, sum);
        }  
    }
    return 0;
}