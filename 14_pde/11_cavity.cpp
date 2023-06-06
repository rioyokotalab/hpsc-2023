#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
using namespace std;
typedef vector<double> arr;
typedef vector<arr> mat;

// #define DEBUG 0

const int M = 1024; // num of threads per block
#define BLOCKS(n_loop) (n_loop + M - 1) / M

__global__ void init_zeros(double *a, int ny, int nx) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = index / ny;
  const int i = index % nx;
  if (j >= ny)
    return;
  if (i >= nx)
    return;
  a[j][i] = 0;
}

int main() {
  const int nx = 161;
  const int ny = 161;
  int nt = 10;
  int nit = 50;
  double dx = 2. / (nx - 1);
  double dy = 2. / (ny - 1);
  double dt = 0.0025;
  double rho = 1;
  double nu = 0.02;

  vector<double> x(nx), y(ny);
  mat u(ny, arr(nx, 0));
  mat v(ny, arr(nx, 0));
  // mat p(ny, arr(nx, 0));
  double *p; // cuda
  mat b(ny, arr(nx, 0));

  // mat pn(ny, arr(nx, 0));
  double *pn; // cuda
  mat un(ny, arr(nx, 0));
  mat vn(ny, arr(ny, 0));

  // allocate cuda-related arrays
  cudaMallocManaged(&p, nx * ny * sizeof(double));
  cudaMallocManaged(&pn, nx * ny * sizeof(double));

  // initialize cuda-related arrays
  init_zeros<<<BLOCKS(ny * nx), M>>>(p, ny, nx);
  init_zeros<<<BLOCKS(ny * nx), M>>>(pn, ny, nx);

  // initialize x and y
  for (int i = 0; i < nx; i++)
    x[i] = i * dx;
  for (int j = 0; j < ny; j++)
    y[j] = j * dy;
#ifdef DEBUG
  chrono::steady_clock::time_point tic, tic_iter;
  chrono::steady_clock::time_point toc, toc_iter;
  double time;
#endif // DEBUG

  // main for, do not apply openmp
  for (int n = 0; n < nt; n++) {
#ifdef DEBUG
    cout << "n:" << n << endl;
    tic = chrono::steady_clock::now();
#endif // DEBUG
    for (int j = 1; j < ny - 1; j++) {
      for (int i = 1; i < nx - 1; i++) {
        b[j][i] = rho * (1 / dt *
                                ((u[j][i + 1] - u[j][i - 1]) / (2 * dx) +
                                    (v[j + 1][i] - v[j - 1][i]) / (2 * dy)) -
                            ((u[j][i + 1] - u[j][i - 1]) / (2 * dx)) *
                                ((u[j][i + 1] - u[j][i - 1]) / (2 * dx)) -
                            2 * ((u[j + 1][i] - u[j - 1][i]) / (2 * dy) *
                                    (v[j][i + 1] - v[j][i - 1]) / (2 * dx)) -
                            ((v[j + 1][i] - v[j - 1][i]) / (2 * dy)) *
                                ((v[j + 1][i] - v[j - 1][i]) / (2 * dy)));
      }
    }
#ifdef DEBUG
    toc = chrono::steady_clock::now();
    time = chrono::duration<double>(toc - tic).count();
    cout << "1st ij loop       : " << time << endl;
#endif // DEBUG
#ifdef DEBUG
    tic = chrono::steady_clock::now();
#endif // DEBUG
    // iteration : do not apply openmp
    for (int it = 0; it < nit; it++) {
#ifdef DEBUG
      tic_iter = chrono::steady_clock::now();
#endif // DEBUG
      // copy p to pn
      // pn = p;  // is this ok?
      for (int j = 1; j < ny - 1; j++) {
        for (int i = 1; i < nx - 1; i++) {
          pn[j][i] = p[j][i];
        }
      }
#ifdef DEBUG
      toc_iter = chrono::steady_clock::now();
      time = chrono::duration<double>(toc_iter - tic_iter).count();
      cout << "iteration loop copy p to pn : " << time << endl;
#endif // DEBUG
#ifdef DEBUG
      tic_iter = chrono::steady_clock::now();
#endif // DEBUG
      // TODO: optimize this loop!!
      for (int j = 1; j < ny - 1; j++) {
        for (int i = 1; i < nx - 1; i++) {
          p[j][i] = (dy * dy * (pn[j][i + 1] + pn[j][i - 1]) +
                        dx * dx * (pn[j + 1][i] + pn[j - 1][i]) -
                        b[j][i] * dx * dx * dy * dy) /
                    (2 * (dx * dx + dy * dy));
        }
      }
#ifdef DEBUG
      toc_iter = chrono::steady_clock::now();
      time = chrono::duration<double>(toc_iter - tic_iter).count();
      cout << "iteration loop update p     : " << time << endl;
#endif // DEBUG

#ifdef DEBUG
      tic_iter = chrono::steady_clock::now();
#endif // DEBUG
      for (int j = 0; j < ny; j++) {
        p[j][nx - 1] = p[j][nx - 2];
        p[j][0] = p[j][1];
      }
      for (int i = 0; i < nx; i++) {
        p[0][i] = p[1][i];
        p[ny - 1][i] = 0;
      }
#ifdef DEBUG
      toc_iter = chrono::steady_clock::now();
      time = chrono::duration<double>(toc_iter - tic_iter).count();
      cout << "iteration loop boundary of p: " << time << endl;
#endif // DEBUG
    }
#ifdef DEBUG
    toc = chrono::steady_clock::now();
    time = chrono::duration<double>(toc - tic).count();
    cout << "iteration loop    : " << time << endl;
#endif // DEBUG

#ifdef DEBUG
    tic = chrono::steady_clock::now();
#endif // DEBUG
    // un = u.copy()
    // vn = v.copy()
    for (int j = 1; j < ny - 1; j++) {
      for (int i = 1; i < nx - 1; i++) {
        un[j][i] = u[j][i];
        vn[j][i] = v[j][i];
      }
    }
#ifdef DEBUG
    toc = chrono::steady_clock::now();
    time = chrono::duration<double>(toc - tic).count();
    cout << "copy loop (u,v)   : " << time << endl;
#endif // DEBUG
#ifdef DEBUG
    tic = chrono::steady_clock::now();
#endif // DEBUG
    for (int j = 1; j < ny - 1; j++) {
      for (int i = 1; i < nx - 1; i++) {
        u[j][i] = (un[j][i] - un[j][i] * dt / dx * (un[j][i] - un[j][i - 1]) -
                   un[j][i] * dt / dy * (un[j][i] - un[j - 1][i]) -
                   dt / (2 * rho * dx) * (p[j][i + 1] - p[j][i - 1]) +
                   nu * dt / dx * dx *
                       (un[j][i + 1] - 2 * un[j][i] + un[j][i - 1]) +
                   nu * dt / dy * dy *
                       (un[j + 1][i] - 2 * un[j][i] + un[j - 1][i]));
        v[j][i] = (vn[j][i] - vn[j][i] * dt / dx * (vn[j][i] - vn[j][i - 1]) -
                   vn[j][i] * dt / dy * (vn[j][i] - vn[j - 1][i]) -
                   dt / (2 * rho * dx) * (p[j + 1][i] - p[j - 1][i]) +
                   nu * dt / dx * dx *
                       (vn[j][i + 1] - 2 * vn[j][i] + vn[j][i - 1]) +
                   nu * dt / dy * dy *
                       (vn[j + 1][i] - 2 * vn[j][i] + vn[j - 1][i]));
      }
    }
#ifdef DEBUG
    toc = chrono::steady_clock::now();
    time = chrono::duration<double>(toc - tic).count();
    cout << "u,v update loop   : " << time << endl;
#endif // DEBUG
#ifdef DEBUG
    tic = chrono::steady_clock::now();
#endif // DEBUG
    for (int j = 0; j < ny; j++) {
      u[j][0] = 0;
      u[j][nx - 1] = 0;
      v[j][0] = 0;
      v[j][nx - 1] = 0;
    }
    for (int i = 0; i < nx; i++) {
      u[0][i] = 0;
      u[ny - 1][i] = 1;
      v[0][i] = 0;
      v[ny - 1][i] = 0;
    }
#ifdef DEBUG
    toc = chrono::steady_clock::now();
    time = chrono::duration<double>(toc - tic).count();
    cout << "boundary condition: " << time << endl;
#endif // DEBUG
  }

  return 0;
}
