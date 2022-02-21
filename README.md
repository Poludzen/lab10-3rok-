# Problem - numerical integration by the method of rectangles, trapezoids and Simpson.
Function -> f (x) = x * x, [-4.2] result = 24 \
Comparison of the execution times of the algorithms on the CPU and GPU. 
## CPU 
![image](https://user-images.githubusercontent.com/80580379/154992695-c5aa26f8-5c97-4014-bd96-c100970ebb5a.png)
### Conclusions:
1. Rectangles Method: for small n the runtimes are quite short, but the accuracy of the calculations is low; long execution time for large n, but with greater accuracy.
2. Trapezoidal method: corrects the situation with an accuracy for large n and reduce times.
3. Simpson's method: best method for small n; we already have a good answer with n = 10 and in a short time.
## GPU
![image](https://user-images.githubusercontent.com/80580379/154992740-f256feef-cfa7-4bef-9dd0-db334bf52a6e.png)

### Conclusions:
1. Rectangles Method: for small n the runtime is longer than on the CPU, and the accuracy of the calculation is low; for larger n the execution times on the GPU are slower than on the CPU.
2. Trapezoidal Method: corrects the situation with an accuracy for large n and shortens times a bit.
3. Simpson's method: best method for small n and for CPU; we already have a good answer, but the execution time on the GPU is greater than on the CPU.
## Summary:
The GPU version turned out to be less efficient than the small n version of the CPU. \
GPU acceleration increases as n increases. \
The GPU code is a heavy overhead when calling 'malloc' to allocate memory arrays and copy data between the CPU and GPU. \
CPU code doesn't use any arrays, so it avoids this overhead. It follows that the execution times are shorter. \
But for larger values of n, the GPU code version turned out to be more efficient than the CPU.
