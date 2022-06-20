#include <starpu.h>

void cpu_func(void *buffers[], void *) {
  float *a = (float *)STARPU_VECTOR_GET_PTR(buffers[0]);
  int N = (int)STARPU_VECTOR_GET_NX(buffers[0]);
  for(int i=0; i<N; i++)
    a[i] = i;
}

static __global__ void cuda_kernel(float *a) {
  int i = threadIdx.x;
  a[i] = -i;
}

void cuda_func(void *buffers[], void *) {
  float *a = (float *)STARPU_VECTOR_GET_PTR(buffers[0]);
  int N = STARPU_VECTOR_GET_NX(buffers[0]);
  cuda_kernel<<<1,N,0,starpu_cuda_get_local_stream()>>>(a);
  cudaStreamSynchronize(starpu_cuda_get_local_stream());
}

int main() {
  const int N=8;
  float a[N];
  int ret = starpu_init(NULL);
  starpu_data_handle_t vector_handle;
  starpu_vector_data_register(&vector_handle,0,(uintptr_t)a,N,sizeof(float));
  starpu_codelet cl;
  starpu_codelet_init(&cl);
  cl.nbuffers = 1;
  cl.where = STARPU_CPU;
  cl.cpu_funcs[0] = cpu_func;
  starpu_task_insert(&cl,STARPU_RW,vector_handle,0);
  starpu_task_wait_for_all();
  for(int i=0; i<N; i++)
    printf("%d %g\n",i,a[i]);
  cl.where = STARPU_CUDA;
  cl.cuda_funcs[0] = cuda_func;
  starpu_task_insert(&cl,STARPU_RW,vector_handle,0);
  starpu_task_wait_for_all();
  starpu_data_unregister(vector_handle);
  for(int i=0; i<N; i++)
    printf("%d %g\n",i,a[i]);
  starpu_shutdown();
}
