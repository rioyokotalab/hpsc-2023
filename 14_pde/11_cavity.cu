#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
using namespace std;
typedef vector<double> arr;
typedef vector<arr> mat;

#define DEBUG 0
// #define DEBUG_ITER 0

const int M = 1024; // num of threads per block
#define BLOCKS(n_loop) (n_loop + M - 1) / M

void matalloc(double **ptr, int ny, int nx) {
  cudaMallocManaged(ptr, ny * nx * sizeof(double));
}

__global__ void init_zeros(double **a, int ny, int nx) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = index / ny;
  const int i = index % nx;
  if (j >= ny)
    return;
  if (i >= nx)
    return;
  a[(j)*nx + i] = 0;
}

__global__ void matcopy(double **src, double **dst, int ny, int nx) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = index / ny;
  const int i = index % nx;
  if (!(1 <= j && j < ny - 1))
    return;
  if (!(1 <= i && i < nx - 1))
    return;
  dst[(j)*nx + i] = src[(j)*nx + i];
}

__global__ void update_b(double **b, double **u, double **v, int dt, int dy,
    int dx, int ny, int nx, double rho) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = index / ny;
  const int i = index % nx;
  if (!(1 <= j && j < ny - 1))
    return;
  if (!(1 <= i && i < nx - 1))
    return;
  b[(j)*nx + i] =
      rho *
      (1 / dt *
              ((u[(j)*nx + i + 1] - u[(j)*nx + i - 1]) / (2 * dx) +
                  (v[(j + 1) * nx + i] - v[(j - 1) * nx + i]) / (2 * dy)) -
          ((u[(j)*nx + i + 1] - u[(j)*nx + i - 1]) / (2 * dx)) *
              ((u[(j)*nx + i + 1] - u[(j)*nx + i - 1]) / (2 * dx)) -
          2 * ((u[(j + 1) * nx + i] - u[(j - 1) * nx + i]) / (2 * dy) *
                  (v[(j)*nx + i + 1] - v[(j)*nx + i - 1]) / (2 * dx)) -
          ((v[(j + 1) * nx + i] - v[(j - 1) * nx + i]) / (2 * dy)) *
              ((v[(j + 1) * nx + i] - v[(j - 1) * nx + i]) / (2 * dy)));
}

__global__ void update_p(
    double **p, double **pn, double **b, int dy, int dx, int ny, int nx) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = index / ny;
  const int i = index % nx;
  if (!(1 <= j && j < ny - 1))
    return;
  if (!(1 <= i && i < nx - 1))
    return;
  p[(j)*nx + i] = (dy * dy * (pn[(j)*nx + i + 1] + pn[(j)*nx + i - 1]) +
                      dx * dx * (pn[(j + 1) * nx + i] + pn[(j - 1) * nx + i]) -
                      b[(j)*nx + i] * dx * dx * dy * dy) /
                  (2 * (dx * dx + dy * dy));
}

__global__ void boundary_p_y(double **p, int ny, int nx) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= ny)
    return;
  p[(j)*nx + nx - 1] = p[(j)*nx + nx - 2];
  p[(j)*nx + 0] = p[(j)*nx + 1];
}
__global__ void boundary_p_x(double **p, int ny, int nx) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nx)
    return;
  p[(0) * nx + i] = p[(1) * nx + i];
  p[(ny - 1) * nx + i] = 0;
}

__global__ void update_u(double **u, double **un, double **p, int dt, int dy,
    int dx, int ny, int nx, double rho, double nu) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = index / ny;
  const int i = index % nx;
  if (!(1 <= j && j < ny - 1))
    return;
  if (!(1 <= i && i < nx - 1))
    return;
  u[(j)*nx + i] =
      (un[(j)*nx + i] -
          un[(j)*nx + i] * dt / dx * (un[(j)*nx + i] - un[(j)*nx + i - 1]) -
          un[(j)*nx + i] * dt / dy * (un[(j)*nx + i] - un[(j - 1) * nx + i]) -
          dt / (2 * rho * dx) * (p[(j)*nx + i + 1] - p[(j)*nx + i - 1]) +
          nu * dt / dx * dx *
              (un[(j)*nx + i + 1] - 2 * un[(j)*nx + i] + un[(j)*nx + i - 1]) +
          nu * dt / dy * dy *
              (un[(j + 1) * nx + i] - 2 * un[(j)*nx + i] +
                  un[(j - 1) * nx + i]));
}

__global__ void update_v(double **v, double **vn, double **p, int dt, int dy,
    int dx, int ny, int nx, double rho, double nu) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = index / ny;
  const int i = index % nx;
  if (!(1 <= j && j < ny - 1))
    return;
  if (!(1 <= i && i < nx - 1))
    return;
  v[(j)*nx + i] =
      (vn[(j)*nx + i] -
          vn[(j)*nx + i] * dt / dx * (vn[(j)*nx + i] - vn[(j)*nx + i - 1]) -
          vn[(j)*nx + i] * dt / dy * (vn[(j)*nx + i] - vn[(j - 1) * nx + i]) -
          dt / (2 * rho * dx) * (p[(j + 1) * nx + i] - p[(j - 1) * nx + i]) +
          nu * dt / dx * dx *
              (vn[(j)*nx + i + 1] - 2 * vn[(j)*nx + i] + vn[(j)*nx + i - 1]) +
          nu * dt / dy * dy *
              (vn[(j + 1) * nx + i] - 2 * vn[(j)*nx + i] +
                  vn[(j - 1) * nx + i]));
}

__global__ void boundary_u_v_y(double **u, double **v, int ny, int nx) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= ny)
    return;
  u[(j)*nx + 0] = 0;
  u[(j)*nx + nx - 1] = 0;
  v[(j)*nx + 0] = 0;
  v[(j)*nx + nx - 1] = 0;
}

__global__ void boundary_u_v_x(double **u, double **v, int ny, int nx) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nx)
    return;
  u[(0) * nx + i] = 0;
  u[(ny - 1) * nx + i] = 1;
  v[(0) * nx + i] = 0;
  v[(ny - 1) * nx + i] = 0;
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

  double *u;
  double *v;
  double *p; // cuda
  double *b; // cuda

  double *pn; // cuda
  double *un;
  double *vn;

#ifdef DEBUG
  chrono::steady_clock::time_point tic;
  chrono::steady_clock::time_point toc;
  double time;
#endif // DEBUG
#ifdef DEBUG_ITER
  chrono::steady_clock::time_point tic_iter;
  chrono::steady_clock::time_point toc_iter;
  double time_iter;
#endif // DEBUG_ITER

#ifdef DEBUG
  tic = chrono::steady_clock::now();
#endif // DEBUG

  // allocate cuda-related arrays
  matalloc(&p, ny, nx);
  matalloc(&pn, ny, nx);
  matalloc(&b, ny, nx);
  matalloc(&u, ny, nx);
  matalloc(&v, ny, nx);
  matalloc(&un, ny, nx);
  matalloc(&vn, ny, nx);

  // initialize cuda-related mat
  init_zeros<<<BLOCKS(ny * nx), M>>>(p, ny, nx);
  init_zeros<<<BLOCKS(ny * nx), M>>>(pn, ny, nx);
  init_zeros<<<BLOCKS(ny * nx), M>>>(b, ny, nx);
  init_zeros<<<BLOCKS(ny * nx), M>>>(u, ny, nx);
  init_zeros<<<BLOCKS(ny * nx), M>>>(v, ny, nx);
  init_zeros<<<BLOCKS(ny * nx), M>>>(un, ny, nx);
  init_zeros<<<BLOCKS(ny * nx), M>>>(vn, ny, nx);
  cudaDeviceSynchronize();

#ifdef DEBUG
  toc = chrono::steady_clock::now();
  time = chrono::duration<double>(toc - tic).count();
  cout << "allocation-related: " << time << endl;
#endif // DEBUG

  // main for, do not apply openmp
  for (int n = 0; n < nt; n++) {
#ifdef DEBUG
    cout << "n:" << n << endl;
    tic = chrono::steady_clock::now();
#endif // DEBUG
    update_b<<<BLOCKS(nx * ny), M>>>(b, u, v, dt, dy, dx, ny, nx, rho);
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
#ifdef DEBUG_ITER
      tic_iter = chrono::steady_clock::now();
#endif // DEBUG_ITER
      // copy p to pn
      matcopy<<<BLOCKS(nx * ny), M>>>(p, pn, ny, nx);
      cudaDeviceSynchronize();
#ifdef DEBUG_ITER
      toc_iter = chrono::steady_clock::now();
      time_iter = chrono::duration<double>(toc_iter - tic_iter).count();
      cout << "iteration loop copy p to pn : " << time_iter << endl;
#endif // DEBUG_ITER
#ifdef DEBUG_ITER
      tic_iter = chrono::steady_clock::now();
#endif // DEBUG_ITER
      // }
      update_p<<<BLOCKS(nx * ny), M>>>(p, pn, b, dy, dx, ny, nx);
      cudaDeviceSynchronize();
#ifdef DEBUG_ITER
      toc_iter = chrono::steady_clock::now();
      time_iter = chrono::duration<double>(toc_iter - tic_iter).count();
      cout << "iteration loop update p     : " << time_iter << endl;
#endif // DEBUG_ITER

#ifdef DEBUG_ITER
      tic_iter = chrono::steady_clock::now();
#endif // DEBUG_ITER
      boundary_p_y<<<BLOCKS(nx * ny), M>>>(p, ny, nx);
      boundary_p_x<<<BLOCKS(nx * ny), M>>>(p, ny, nx);
      cudaDeviceSynchronize();
#ifdef DEBUG_ITER
      toc_iter = chrono::steady_clock::now();
      time_iter = chrono::duration<double>(toc_iter - tic_iter).count();
      cout << "iteration loop boundary of p: " << time_iter << endl;
#endif // DEBUG_ITER
    }
#ifdef DEBUG
    toc = chrono::steady_clock::now();
    time = chrono::duration<double>(toc - tic).count();
    cout << "iteration loop    : " << time << endl;
#endif // DEBUG

#ifdef DEBUG
    tic = chrono::steady_clock::now();
#endif // DEBUG
    matcopy<<<BLOCKS(nx * ny), M>>>(u, un, ny, nx);
    matcopy<<<BLOCKS(nx * ny), M>>>(v, vn, ny, nx);
    cudaDeviceSynchronize();
#ifdef DEBUG
    toc = chrono::steady_clock::now();
    time = chrono::duration<double>(toc - tic).count();
    cout << "copy loop (u,v)   : " << time << endl;
#endif // DEBUG
#ifdef DEBUG
    tic = chrono::steady_clock::now();
#endif // DEBUG
    update_u<<<BLOCKS(nx * ny), M>>>(u, un, p, dt, dy, dx, ny, nx, rho, nu);
    update_v<<<BLOCKS(nx * ny), M>>>(v, vn, p, dt, dy, dx, ny, nx, rho, nu);
    cudaDeviceSynchronize();
#ifdef DEBUG
    toc = chrono::steady_clock::now();
    time = chrono::duration<double>(toc - tic).count();
    cout << "u,v update loop   : " << time << endl;
#endif // DEBUG
#ifdef DEBUG
    tic = chrono::steady_clock::now();
#endif // DEBUG
    boundary_u_v_y<<<BLOCKS(nx * ny), M>>>(u, v, ny, nx);
    boundary_u_v_x<<<BLOCKS(nx * ny), M>>>(u, v, ny, nx);
    cudaDeviceSynchronize();
#ifdef DEBUG
    toc = chrono::steady_clock::now();
    time = chrono::duration<double>(toc - tic).count();
    cout << "boundary condition: " << time << endl;
#endif // DEBUG
  }

  return 0;
}
