#include <cstdio>
#include <cassert>
#include "xsimd/xsimd.hpp"

int main() {
  const int N = 8;
  xsimd::batch<float, xsimd::avx> a = {1, 1, 1, 1, 1, 1, 1, 1};
  float b = xsimd::hadd(a);
  printf("%g\n",b);
}
