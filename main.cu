#include <iostream>

__device__ __host__  double f(double  d){
    return d * d;
}

__global__ void trapezoidIntegral(const double* a, const double* b, const int* n, double* out){
    const int ind = blockIdx.x * blockDim.x + threadIdx.x;
    const double width = (b[ind]-a[ind])/n[ind];

    double trapezoidal_integral = 0;
    for(int step = 0; step < n[ind]; step++) {
        const double x1 = a[ind] + step*width;
        const double x2 = a[ind] + (step+1)*width;

        trapezoidal_integral += 0.5*(x2-x1)*(f(x1) + f(x2));
    }

    out[ind] = trapezoidal_integral;
}
__global__ void simpsonIntegral(const double* a, const double* b, const int* n, double* out){
    const int ind = blockIdx.x * blockDim.x + threadIdx.x;
    const double width = (b[ind]-a[ind])/n[ind];

    double simpson_integral = 0;
    for(int step = 0; step < n[ind]; step++) {
        const double x1 = a[ind] + step*width;
        const double x2 = a[ind] + (step+1)*width;

        simpson_integral += (x2-x1)/6.0*(f(x1) + 4.0*f(0.5*(x1+x2)) + f(x2));
    }

    out[ind] = simpson_integral;
}
__global__ void pramougIntegral(const double* a, const double* b, const int* n, double* out){
    const int ind = blockIdx.x * blockDim.x + threadIdx.x;
    double x, h;
    double sum = 0.0;
    double fx;
    h = (b[ind] - a[ind]) / n[ind];  //шаг

    for (int i = 0; i < n[ind]; i++) {
        x = a[ind] + i * h;
        fx = f(x);
        sum += fx;
    }
    out[ind] =  (sum * h);
}

double trapezoidIntegral(const double a, const double b, const int n){
    const double width = (b - a) / n;

    double trapezoidal_integral = 0;
    for(int step = 0; step < n; step++) {
        const double x1 = a + step * width;
        const double x2 = a + (step + 1) * width;

        trapezoidal_integral += 0.5*(x2-x1)*(f(x1) + f(x2));
    }

    return trapezoidal_integral;
}
double simpsonIntegral(const double a, const double b, const int n){
    const double width = (b-a)/n;

    double simpson_integral = 0;
    for(int step = 0; step < n; step++) {
        const double x1 = a + step*width;
        const double x2 = a + (step+1)*width;

        simpson_integral += (x2-x1)/6.0*(f(x1) + 4.0*f(0.5*(x1+x2)) + f(x2));
    }

    return simpson_integral;
}
double pramougIntegral(const double a, const double b, const int n){
    double x, h;
    double sum = 0.0;
    double fx;
    h = (b - a) / n;  //шаг

    for (int i = 0; i < n; i++) {
        x = a + i * h;
        fx = f(x);
        sum += fx;
    }
    return sum * h;
}


double* IntegralTestCPU(int count, double a_, double b_, int n_){
    auto* arr = (double *)malloc(sizeof(double) * count);
    for (int i = 0; i < count; i++){
        arr[i] = pramougIntegral(a_,b_,n_);
    }
    return arr;
}
double* IntegralTestGPU(int count, double a_, double b_, int n_){
    double *a, *b, *out;
    int *n;
    cudaMalloc((void **)&a,count * sizeof(double));
    cudaMalloc((void **)&b,count * sizeof(double));
    cudaMalloc((void **)&n,count * sizeof(int));
    cudaMalloc((void**)&out, sizeof(double)* count);

    auto * a_new = (double *)malloc(sizeof(double ) * count);
    auto * b_new = (double *)malloc(sizeof(double ) * count);
    int * n_new = (int *)malloc(sizeof(int ) * count);

    for (int i = 0; i < count; i++){
        a_new[i] = a_;
        b_new[i] = b_;
        n_new[i] = n_;
    }

    cudaMemcpy(a, a_new, sizeof(double ) * count, cudaMemcpyHostToDevice);
    cudaMemcpy(b, b_new, sizeof(double ) * count, cudaMemcpyHostToDevice);
    cudaMemcpy(n, n_new, sizeof(int) * count, cudaMemcpyHostToDevice);

    pramougIntegral<<<count / 1000 + 1, 1000>>>(a, b, n, out);

    auto * array_copy = new double [count];
    cudaMemcpy(array_copy,out,count* sizeof(double ),cudaMemcpyDeviceToHost);
    return array_copy;
}


int main() {
    time_t  start = clock();
    double* a = IntegralTestCPU(100000, 1, 1000, 100000);
    time_t end = clock();
    double seconds = (double)(end - start) / CLOCKS_PER_SEC;
    printf("The time: %f seconds\n", seconds);



    time_t  start1 = clock();
    double* a1 = IntegralTestGPU(100000, 1, 1000, 100000);
    time_t end1 = clock();
    double seconds1 = (double)(end1 - start1) / CLOCKS_PER_SEC;
    printf("The time: %f seconds\n", seconds1);

    return 0;
}
