#include <cstdio>
#include <starpu.h>

void cpu_func(void **, void *) {
  printf("hello\n");
}

int main() {
  int err = starpu_init(NULL);
  starpu_codelet cl;
  starpu_codelet_init(&cl);
  cl.cpu_funcs[0] = cpu_func;
  starpu_task_insert(&cl,0);
  starpu_task_wait_for_all();
  starpu_shutdown();
}
