#include "cuda_def.h"

#include <stdlib.h>
#include <stdio.h>

namespace global_gauss_2d {

	#define DEBUG_ENABLED false

	#define DEBUG(code) if (DEBUG_ENABLED) code;

	#define ELEMENT(a, h, i, j) a[i+j*h]

	#define BLOCK_WIDTH  8
	#define BLOCK_HEIGHT 16

	#define THREAD_WIDTH  4
	#define THREAD_HEIGHT 2

	/*
	  ____________________________________________
	 |                                            |
	 |                                            |
	 |       _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _|
	 |      !c |______________const_______________|
	 |      !o |_ _ :_ _|        |       |        |
	 |      !n |____:___|________|_______|________|
	 |      !s |        |        |       |        |
	 |		!t |________|________|_______|________|
	 |      !  |        |        |       |        |
	 |______!__|________|________|_______|________|

	*/

	__global__ void reduction_kernel(DeviceSystem<float> a, int k, float* multimpliers);
	__global__ void mul_preparation_kernel(DeviceSystem<float> a, int k, float* multimpliers);
	void solve(DeviceSystem<float>& a);

	extern "C" void global_gauss_2d(System<float>& system) {
		size_t system_size = sizeof(float)*system.dim()*(system.dim()+1);
		DeviceSystem<float> dev_system((dev_size)system.dim());
		assertSuccess(cudaSetDevice(0), "cudaSetDevice(0) failed.");
		assertSuccess(cudaMalloc(&dev_system.arr, system_size), "cudaMalloc for device system failed.");
		assertSuccess(cudaMemcpy(dev_system.arr, system.array, system_size, cudaMemcpyHostToDevice), "cudaMemcpy failed to copy system form hoste to device.");
		CUDA_TIME_OF(solve(dev_system));

		assertSuccess(cudaMemcpy(system.array, dev_system.arr, system_size, cudaMemcpyDeviceToHost), "cudaMemcpy failed to copy system form device to host.");
		assertSuccess(cudaFree(dev_system.arr), "cudaFree failed to free device memory of system.");
		assertSuccess(cudaDeviceReset(), "cudaDeviceReset() deiled.");
	}

	void solve(DeviceSystem<float>& system) {
		int N = system.dim;
		float* dev_multimpliers;
		assertSuccess(cudaMalloc(&dev_multimpliers, sizeof(float)*N));
		for (int k=0; k<N-1; k++) {
			const int SUB_MATRIX_WIDTH  = (N + 1) - k;		// martix width - numbre of already prepared columns - current colum (it is definitly will be 0 vector)
			const int SUB_MATRIX_HEIGHT = N - (k + 1);     // matrix height - number of already prepared rows
			const int GRID_WIDTH  = div_ceiling(SUB_MATRIX_WIDTH, BLOCK_WIDTH * THREAD_WIDTH);
			const int GRID_HEIGHT = div_ceiling(SUB_MATRIX_HEIGHT, BLOCK_HEIGHT * THREAD_HEIGHT);
			dim3 blocks(GRID_WIDTH, GRID_HEIGHT);
			dim3 threads(BLOCK_WIDTH, BLOCK_HEIGHT);
			mul_preparation_kernel<<<1, SUB_MATRIX_HEIGHT>>>(system, k, dev_multimpliers);
			reduction_kernel<<<blocks, threads>>>(system, k, dev_multimpliers);
		}
		assertSuccess(cudaFree(dev_multimpliers));
	}

	__global__ void mul_preparation_kernel(DeviceSystem<float> a, int k, float* multimpliers) {
		int i = threadIdx.x + k + 1;
		multimpliers[i] = a[i][k] / a[k][k];
	}

	__global__ void reduction_kernel(DeviceSystem<float> a, int k, float* multimpliers) {
		int N = a.dim;
		int i_gl = (blockIdx.y * BLOCK_HEIGHT + threadIdx.y) * THREAD_HEIGHT + (k + 1);
		int j_gl = (blockIdx.x * BLOCK_WIDTH  + threadIdx.x) * THREAD_WIDTH  + (k);
		for (int i = i_gl; i < min(i_gl + THREAD_HEIGHT, N); i++) {
			float multiplier = multimpliers[i];
			for(int j = j_gl; j < min(j_gl + THREAD_WIDTH, N+1); j++) {
				a[i][j] = a[i][j] - multiplier * a[k][j];
			}
		}
	}

}