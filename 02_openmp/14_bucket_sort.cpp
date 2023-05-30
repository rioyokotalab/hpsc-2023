#include <cstdio>
#include <cstdlib>
#include <vector>
#include <omp.h>

int main() {
  int n = 50;
  int range = 5;
  std::vector<int> key(n);
  for (int i = 0; i < n; i++) {
    key[i] = rand() % range;
    printf("%d ", key[i]);
  }
  printf("\n");

  std::vector<int> bucket(range, 0);
  #pragma omp parallel for
  for (int i = 0; i < n; i++) {
    #pragma omp atomic
    bucket[key[i]]++;
  }

  std::vector<int> offset(range, 0);
  int sum = 0;
  for (int i = 0; i < range; i++) {
    offset[i] = sum;
    sum += bucket[i];
  }

  std::vector<int> sorted_key(n);
  #pragma omp parallel for
  for (int i = 0; i < n; i++) {
    int index = offset[key[i]];
    sorted_key[index] = key[i];
    #pragma omp atomic
    offset[key[i]]++;
  }

  for (int i = 0; i < n; i++) {
    printf("%d ", sorted_key[i]);
  }
  printf("\n");
}
