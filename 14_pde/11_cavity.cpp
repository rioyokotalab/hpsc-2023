#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
using namespace std;

typedef vector<double> arr;
typedef vector<arr> mat;

#define DEBUG 0

int main() {
  const int nx = 41;
  const int ny = 41;
  int nt = 500;
  int nit = 50;
  double dx = 2 / (nx - 1);
  double dy = 2 / (ny - 1);
  double dt = 0.01;
  double rho = 1;
  double nu = 0.02;

  vector<double> x(nx), y(ny);
  mat u(ny, arr(nx, 0));
  mat v(ny, arr(nx, 0));
  mat p(ny, arr(nx, 0));
  mat b(ny, arr(nx, 0));

  mat pn(ny, arr(nx, 0));
  mat un(ny);
  mat vn(ny);

  // initialize x and y
  for (int i = 0; i < nx; i++)
    x[i] = i * dx;
  for (int j = 0; j < ny; j++)
    y[j] = j * dy;

  // main for
  for (int n = 0; n < nt; n++) {
#ifdef DEBUG
    cout << "n:" << n << endl;
#endif
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
    cout << "before nit for" << endl;
#endif
    for (int it = 0; it < nit; nit++) {
      // copy p to pn
      // pn = p;  // is this ok?
#ifdef DEBUG
      cout << "copy p to pn" << endl;
#endif
      for (int j = 1; j < ny - 1; j++) {
        for (int i = 1; i < nx - 1; i++) {
          pn[j][i] = p[j][i];
        }
      }
#ifdef DEBUG
      cout << "calc1" << endl;
#endif
      for (int j = 1; j < ny - 1; j++) {
        for (int i = 1; i < nx - 1; i++) {
          p[j][i] = (dy * dy * (pn[j][i + 1] + pn[j][i - 1]) +
                        dx * dx * (pn[j + 1][i] + pn[j - 1][i]) -
                        b[j][i] * dx * dx * dy * dy) /
                    (2 * (dx * dx + dy * dy));
        }
      }
#ifdef DEBUG
      cout << "Done: calc1" << endl;
#endif

      for (int j = 0; j < ny; j++) {
        p[j][nx - 1] = p[j][nx - 2];
        p[j][0] = p[j][1];
      }
      for (int i = 0; i < nx; i++) {
        p[0][i] = p[1][i];
        p[ny - 1][i] = 0;
      }
    }
#ifdef DEBUG
    cout << "after nit for" << endl;
#endif

    // un = u.copy()
    // vn = v.copy()
#ifdef DEBUG
    cout << "before copy of u and v" << endl;
#endif
    for (int j = 1; j < ny - 1; j++) {
      for (int i = 1; i < nx - 1; i++) {
        un[j][i] = u[j][i];
        vn[j][i] = v[j][i];
      }
    }
#ifdef DEBUG
    cout << "after copy of u and v" << endl;
#endif
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
    for (int j = 0; j < ny; j++) {
      u[j][0] = 0;
      u[j][-1] = 0;
      v[j][0] = 0;
      v[j][-1] = 0;
    }
    for (int i = 0; i < nx; i++) {
      u[0][i] = 0;
      u[-1][i] = 1;
      v[0][i] = 0;
      v[-1][i] = 0;
    }
  }

  return 0;
}
