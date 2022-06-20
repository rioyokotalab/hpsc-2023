#include <cstdio>
#include <cmath>
#include <cassert>
#include "xsimd/xsimd.hpp"

int main() {
  const int N = 8;
  float aa[8];
  xsimd::batch<float, xsimd::avx> b, a = {0, 1, 2, 3, 4, 5, 6, 7};
  b = sin(a);
  for(int i=0; i<N; i++)
    printf("%g %g\n",a.get(i),b.get(i));
}
