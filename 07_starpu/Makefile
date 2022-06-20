00_hello: 00_hello.cpp
	g++ 00_hello.cpp -lstarpu-1.3
01_values: 01_values.cpp
	g++ 01_values.cpp -lstarpu-1.3
02_vectors: 02_vectors.cpp
	g++ 02_vectors.cpp -lstarpu-1.3
03_gpu: 03_gpu.cu
	nvcc 03_gpu.cu -lstarpu-1.3
04_both: 04_both.cu
	nvcc 04_both.cu -lstarpu-1.3
05_lu: 05_lu.cpp
	g++ 05_lu.cpp -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp
06_2x2_lu: 06_2x2_lu.cpp
	g++ 06_2x2_lu.cpp -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp
07_MxM_lu: 07_MxM_lu.cpp
	g++ 07_MxM_lu.cpp -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp
08_openmp_lu: 08_openmp_lu.cpp
	g++ 08_openmp_lu.cpp -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -fopenmp
09_starpu_lu: 09_starpu_lu.cpp
	g++ 09_starpu_lu.cpp -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lstarpu-1.3
