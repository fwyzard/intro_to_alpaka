/*
 * g++ -std=c++20 -O2 -g 00_hello_world.cc -o 00_hello_world
 */

#include <cmath>
#include <cstdio>

void print_sqrt(double x) {
  printf("The square root of %g is %g\n", x, std::sqrt(x));
}

int main() {
  print_sqrt(2.);
}
