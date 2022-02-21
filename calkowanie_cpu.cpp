%%writefile calk_cpu.cpp
#include <iostream>
#include <cstdlib>
#include<stdio.h>

/*
%%shell
c++ calk_cpu.cpp -o cpu
./cpu
*/

using namespace std;

float function1(float x) { return x*x; }

float cpu_prostokaty(float xp, float xk, int n){
    float h, calka = 0;
    h = (xk - xp) / (float)n;
    for (int i = 1; i <= n; i++)
    {
        calka += function1(xp + i*h)*h;
    }
    return calka;
}

float cpu_trapezy(float xp, float xk, int n){
    float h, calka = 0;
    h = (xk - xp) / (float)n;
    for (int i = 1; i < n; i++)
    {
        calka += function1(xp + i * h);
    }
    calka += function1(xp) / 2;
    calka += function1(xk) / 2;
    calka *= h;
    return calka;
}

float cpu_Simpson(float xp, float xk, int n){
    float h, x, calka = 0;
    float st = 0;
    h = (xk - xp) / (float)n;
    for(int i = 1; i <= n; i++)
    {
      x = xp + i * h;
      st += function1(x - h/2);
      if(i < n) calka += function1(x);
    }
    calka = h / 6*(function1(xp) + function1(xk) + 2*calka + 4*st);
    return calka;
}

int main()
{
    clock_t start, end;
    float xp, xk;
 
    xp = -4;
    xk = 2;
 
    for(int n = 10; n <= 100000; n*=10){
        
        start = clock(); 
        cout << "Wynik: " <<  cpu_prostokaty(xp, xk, n) << endl;
        end = clock();
        double cpu_prostok = ((double)(end-start))/CLOCKS_PER_SEC; 
        printf("CPU prostokaty n=%d: %lf sekund\n", n, cpu_prostok);
        cout << "\n"<< endl;
        
        start = clock(); 
        cout << "Wynik: " <<  cpu_trapezy(xp, xk, n) << endl;
        end = clock();
        double cpu_trapezy = ((double)(end-start))/CLOCKS_PER_SEC; 
        printf("CPU trapezy n=%d: %lf sekund\n", n, cpu_trapezy);
        cout << "\n"<< endl;
        
        start = clock(); 
        cout << "Wynik: " <<  cpu_Simpson(xp, xk, n) << endl;
        end = clock();
        double cpu_simp = ((double)(end-start))/CLOCKS_PER_SEC; 
        printf("CPU simpson n=%d: %lf sekund\n", n, cpu_simp);
        cout << "\n"<< endl;
        
    }
    // Generate time 
    /*
    for(int vers = 3; vers < 4; vers++){
        for(int n = 10; n <= 100000; n*=10){
            printf("dla -> %d\n", n);
            for(int i = 1; i <=20; i++){
                
                if(vers == 1){
                    start = clock(); 
                    int sum = cpu_prostokaty(xp, xk, n);
                    end = clock();
                    double cpu_prostok = ((double)(end-start))/CLOCKS_PER_SEC; 
                    printf("%lf sekund\n", cpu_prostok);
                } else if(vers == 2){
                    start = clock(); 
                    int sum = cpu_trapezy(xp, xk, n);
                    end = clock();
                    double cpu_trapezy = ((double)(end-start))/CLOCKS_PER_SEC; 
                    printf("%lf sekund\n", cpu_trapezy);
                } else if(vers == 3){
                    start = clock(); 
                    int sum = cpu_Simpson(xp, xk, n);
                    end = clock();
                    double cpu_simp = ((double)(end-start))/CLOCKS_PER_SEC; 
                    printf("%lf sekund\n", cpu_simp);
                }
            }
        }    
    }
    */
    return 0;
}
