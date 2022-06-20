#include <cstdio>
#include <starpu.h>

void cpu_func(void *buffers[], void *) {
  int N = STARPU_VECTOR_GET_NX(buffers[0]);
  float *a = (float*)STARPU_VECTOR_GET_PTR(buffers[0]);
  for(int i=0; i<N; i++)
    a[i] = i;
}

int main(void) {
  const int N = 8;
  float a[N];
  int ret = starpu_init(NULL);
  starpu_data_handle_t vector_handle;
  starpu_vector_data_register(&vector_handle,0,(uintptr_t)a,N,sizeof(float));
  starpu_codelet cl;
  starpu_codelet_init(&cl);
  cl.nbuffers = 1;
  cl.cpu_funcs[0] = cpu_func;
  starpu_task_insert(&cl,STARPU_RW,vector_handle,0);
  starpu_task_wait_for_all();
  starpu_data_unregister(vector_handle);
  for(int i=0; i<N; i++)
    printf("%d %g\n",i,a[i]);
  starpu_shutdown();
}
