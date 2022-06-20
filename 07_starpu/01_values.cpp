#include <cstdio>
#include <starpu.h>

void cpu_func(void **, void *args) {
  int i;
  float f;
  starpu_codelet_unpack_args(args, &i, &f, 0);
  printf("%d %g\n",i,f);
}

int main(void) {
  int i = 1;
  float f = 1.1;
  int ret = starpu_init(NULL);
  starpu_codelet cl;
  starpu_codelet_init(&cl);
  cl.cpu_funcs[0] = cpu_func;
  starpu_task_insert(&cl,
		     STARPU_VALUE, &i, sizeof(int),
		     STARPU_VALUE, &f, sizeof(float),
		     0);
  starpu_task_wait_for_all();
  starpu_shutdown();
}
