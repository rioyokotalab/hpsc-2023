#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <vector>

#include <mkl_cblas.h>
#include <mkl_lapacke.h>

using namespace std;

void getrf(vector<double>& A, int N) {
  vector<int> ipiv(N);
  LAPACKE_dgetrf(LAPACK_ROW_MAJOR, N, N, A.data(), N, ipiv.data());
}

void trsm(bool left, bool upper, vector<double>& A, int NA, vector<double>& B, int NB) {
  cblas_dtrsm(CblasRowMajor, left ? CblasLeft : CblasRight, upper ? CblasUpper : CblasLower,
              CblasNoTrans, upper ? CblasNonUnit : CblasUnit, NA, NB, 1.0, A.data(), NA, B.data(), NB);
}

void gemm(vector<double>& A, int NA, vector<double>& B, int NB, vector<double>& C) {
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, NA, NB, NA, -1.0,
              A.data(), NA, B.data(), NB, 1.0, C.data(), NB);
}

int main() {
  int N = 16;
  bool left = true, right = false, upper = true, lower = false;
  vector<double> A00(N*N), A01(N*N), A10(N*N), A11(N*N);
  vector<double> x0(N), x1(N);
  vector<double> b0(N), b1(N);
  vector<int> ipiv(N);
  for (int i=0; i<N; i++) {
    x0[i] = drand48();
    x1[i] = drand48();
  }
  for (int i=0; i<N; i++) {
    b0[i] = b1[i] = 0;
    for (int j=0; j<N; j++) {
      A00[N*i+j] = drand48() + (i == j) * 10;
      A01[N*i+j] = drand48();
      A10[N*i+j] = drand48();
      A11[N*i+j] = drand48() + (i == j) * 10;
      b0[i] += A00[N*i+j] * x0[j] + A01[N*i+j] * x1[j];
      b1[i] += A10[N*i+j] * x0[j] + A11[N*i+j] * x1[j];
    }
  }
  // L00,U00 = lu(A00) (overwrite A00)
  getrf(A00, N);
  // U01 = L00^-1 * A01 (overwrite A01)
  trsm(left, lower, A00, N, A01, N);
  // L10 = A01 * U00^-1 (overwrite A10)
  trsm(right, upper, A00, N, A10, N);
  // A11 -= L10 * U01 (overwrite A11)
  gemm(A10, N, A01, N, A11);
  // L11,U11 = lu(A11) (overwrite A11)
  getrf(A11, N);
  // y0 = L00^-1 b0 (overwrite b0)
  trsm(left, lower, A00, N, b0, 1);
  // b1 -= L10 * y0 (overwrite b1)
  gemm(A10, N, b0, 1, b1);
  // y1 = L11^-1 * b1 (overwrite b1)
  trsm(left, lower, A11, N, b1, 1);
  // x1 = U11^-1 * y1 (overwrite b1)
  trsm(left, upper, A11, N, b1, 1);
  // y0 -= U01 * x1 (overwrite b0)
  gemm(A01, N, b1, 1, b0);
  // x0 = U00^-1 * y0 (overwrite b1)
  trsm(left, upper, A00, N, b0, 1);

  double diff = 0, norm = 0;
  for (int i=0; i<N; i++) {
    diff += (x0[i] - b0[i]) * (x0[i] - b0[i]);
    diff += (x1[i] - b1[i]) * (x1[i] - b1[i]);
    norm += x0[i] * x0[i];
    norm += x1[i] * x1[i];
  }
  printf("Error: %g\n",std::sqrt(diff/norm));
}
