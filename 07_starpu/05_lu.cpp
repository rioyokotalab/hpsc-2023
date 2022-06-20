#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <vector>

#include <mkl_cblas.h>
#include <mkl_lapacke.h>

using namespace std;

int main() {
  int N = 16;
  vector<double> A(N*N);
  vector<double> x(N);
  vector<double> b(N);
  vector<int> ipiv(N);
  for (int i=0; i<N; i++)
    x[i] = drand48();
  for (int i=0; i<N; i++) {
    b[i] = 0;
    for (int j=0; j<N; j++) {
      A[N*i+j] = drand48() + (i == j) * 10;
      b[i] += A[N*i+j] * x[j];
    }
  }
  // L,U = lu(A) (overwrite A)
  LAPACKE_dgetrf(LAPACK_ROW_MAJOR, N, N, A.data(), N, ipiv.data());
  // y = L^-1 * b (overwrite b)
  cblas_dtrsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit, N, 1, 1.0, A.data(), N, b.data(), 1);
  // x = U^-1 * y (overwrite b)
  cblas_dtrsm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, 1, 1.0, A.data(), N, b.data(), 1);
  double diff = 0, norm = 0;
  for (int i=0; i<N; i++) {
    diff += (x[i] - b[i]) * (x[i] - b[i]);
    norm += x[i] * x[i];
  }
  printf("Error: %g\n",std::sqrt(diff/norm));
}
