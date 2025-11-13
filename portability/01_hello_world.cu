/*
 * nvcc -std=c++20 -O2 -g 01_hello_world.cu -o 01_hello_world
 */

#include <cmath>
#include <cstdio>

#include <cuda_runtime.h>

__device__
void print_sqrt(double x) {
  printf("The square root of %g is %g\n", x, std::sqrt(x));
}

__global__
void kernel() {
  print_sqrt(2.);
}

int main() {
  kernel<<<1, 1>>>();
  cudaDeviceSynchronize();
}
