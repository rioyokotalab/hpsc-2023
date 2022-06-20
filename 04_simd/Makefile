00_vecadd: 00_vecadd.cpp
	g++ 00_vecadd.cpp -fopt-info-vec-optimized -march=native -O3
01_reduction: 01_reduction.cpp
	g++ 01_reduction.cpp -fopt-info-vec-optimized -march=native -O3
02_if: 02_if.cpp
	g++ 02_if.cpp -fopt-info-vec-optimized -march=native -O3
03_function: 03_function.cpp
	g++ 03_function.cpp -fopt-info-vec-optimized -march=native -O3
04_permute: 04_permute.cpp
	g++ 04_permute.cpp -fopt-info-vec-optimized -march=native -O3
05_sin: 05_sin.cpp
	g++ 05_sin.cpp -fopt-info-vec-optimized -march=native -O3
06_xsimd_reduction: 06_xsimd_reduction.cpp
	g++ 06_xsimd_reduction.cpp -march=native -Ixsimd/include
07_xsimd_sin: 07_xsimd_sin.cpp
	g++ 07_xsimd_sin.cpp -march=native -Ixsimd/include
08_intrinsic_vecadd: 08_intrinsic_vecadd.cpp
	g++ 08_intrinsic_vecadd.cpp -march=native -O3
09_intrinsic_reduction: 09_intrinsic_reduction.cpp
	g++ 09_intrinsic_reduction.cpp -march=native -O3
10_intrinsic_if: 10_intrinsic_if.cpp
	g++ 10_intrinsic_if.cpp -march=native -O3
11_intrinsic_permute: 11_intrinsic_permute.cpp
	g++ 11_intrinsic_permute.cpp -march=native -O3
12_intrinsic_sin: 12_intrinsic_sin.cpp
	icc 12_intrinsic_sin.cpp -xHost
13_nbody: 13_nbody.cpp
	g++ 13_nbody.cpp -march=native -O3
