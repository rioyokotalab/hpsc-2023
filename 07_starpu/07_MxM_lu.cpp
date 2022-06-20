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
  int M = 3;
  int N = 16;
  bool left = true, right = false, upper = true, lower = false;
  vector<vector<double> > A(M*M);
  vector<vector<double> > x(M);
  vector<vector<double> > b(M);
  vector<int> ipiv(N);
  for (int m=0; m<M*M; m++)
    A[m] = vector<double>(N*N);
  for (int m=0; m<M; m++) {
    x[m] = vector<double>(N);
    b[m] = vector<double>(N);
  }
  for (int m=0; m<M; m++) {
    for (int i=0; i<N; i++) {
      x[m][i] = drand48();
      b[m][i] = 0;
    }
  }
  for (int m=0; m<M; m++) {
    for (int n=0; n<M; n++) {
      for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
          A[M*m+n][N*i+j] = drand48() + (m == n) * (i == j) * 10;
          b[m][i] += A[M*m+n][N*i+j] * x[n][j];
	}
      }
    }
  }
  for (int l=0; l<M; l++) {
    getrf(A[M*l+l], N);
    for (int m=l+1; m<M; m++) {
      trsm(left, lower, A[M*l+l], N, A[M*l+m], N);
      trsm(right, upper, A[M*l+l], N, A[M*m+l], N);
    }
    for (int m=l+1; m<M; m++)
      for (int n=l+1; n<M; n++)
	gemm(A[M*m+l], N, A[M*l+n], N, A[M*m+n]);
  }
  for (int m=0; m<M; m++) {
    for (int n=0; n<m; n++)
      gemm(A[M*m+n], N, b[n], 1, b[m]);
    trsm(left, lower, A[M*m+m], N, b[m], 1);
  }
  for (int m=M-1; m>=0; m--) {
    for (int n=M-1; n>m; n--)
      gemm(A[M*m+n], N, b[n], 1, b[m]);
    trsm(left, upper, A[M*m+m], N, b[m], 1);
  }

  double diff = 0, norm = 0;
  for (int m=0; m<M; m++) {
    for (int i=0; i<N; i++) {
      diff += (x[m][i] - b[m][i]) * (x[m][i] - b[m][i]);
      norm += x[m][i] * x[m][i];
    }
  }
  printf("Error: %g\n",std::sqrt(diff/norm));
}
