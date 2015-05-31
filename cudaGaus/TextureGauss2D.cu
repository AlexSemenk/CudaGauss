#include "cuda_def.h"

#include <stdlib.h>
#include <stdio.h>

#define ELEMENT(a, h, i, j) a[i+j*h]

namespace DeviceTextureGauss {

	#define DEBUG_ENABLED false

	#define DEBUG(code) if (DEBUG_ENABLED) code;

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

	//__constant__ float MASTER_ROW[10000];
	//__constant__ float MASTER_COLUMN[10000];

	__global__ void kernel(float* system, int N, int k);
	void solve(float* dev_system, int N);

	__constant__ float MATER_COLUMN[10000];

	extern "C" void texture_gauss_2d(System<float>& s) {
		float* system = s.array;
		const int N = s.dim();

		float* dev_system;
		assertSuccess(cudaSetDevice(0));
		assertSuccess(cudaMalloc(&dev_system, sizeof(float)*N*(N+1)));
		assertSuccess(cudaMemcpy(dev_system, system, sizeof(float)*N*(N+1), cudaMemcpyHostToDevice));

		CUDA_TIME_OF(solve(dev_system, N));

		assertSuccess(cudaMemcpy(system, dev_system, sizeof(float)*N*(N+1), cudaMemcpyDeviceToHost));
		assertSuccess(cudaFree(dev_system));
		assertSuccess(cudaDeviceReset());
	}

	texture<float, cudaTextureType2D, cudaReadModeElementType> readTexture;

	void solve(float* dev_system, int N) {
		size_t ZERO = 0;
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
		assertSuccess(cudaBindTexture2D(&ZERO, readTexture, dev_system, channelDesc, N+1, N, sizeof(float)*(N+1)), "cudaBindTexture2D failed.");
		for (int k=0; k<N-1; k++) {
			const int SUB_MATRIX_WIDTH  = (N + 1) - k;		// martix width - numbre of already prepared columns - current colum (it is definitly will be 0 vector)
			const int SUB_MATRIX_HEIGHT = N - (k + 1);     // matrix height - number of already prepared rows
			const int GRID_WIDTH  = div_ceiling(SUB_MATRIX_WIDTH, BLOCK_WIDTH * THREAD_WIDTH);
			const int GRID_HEIGHT = div_ceiling(SUB_MATRIX_HEIGHT, BLOCK_HEIGHT * THREAD_HEIGHT);
			dim3 blocks(GRID_WIDTH, GRID_HEIGHT);
			dim3 threads(BLOCK_WIDTH, BLOCK_HEIGHT);
			kernel<<<blocks, threads>>>(dev_system, N, k);
		}
		cudaUnbindTexture(readTexture);
	}

	__global__ void kernel(float* system, int N, int k) {
		int i_gl = (blockIdx.y * BLOCK_HEIGHT + threadIdx.y) * THREAD_HEIGHT + (k + 1);
		int j_gl = (blockIdx.x * BLOCK_WIDTH  + threadIdx.x) * THREAD_WIDTH  + (k);
		for (int i = i_gl; i < min(i_gl + THREAD_HEIGHT, N); i++) {
			float multiplier = ELEMENT(system, N, i, k) / ELEMENT(system, N, k, k);
			for(int j = j_gl; j < min(j_gl + THREAD_WIDTH, N+1); j++) {
				//ELEMENT(system, N, i, j) = ELEMENT(system, N, i, j) - multiplier * ELEMENT(system, N, k, j);
				ELEMENT(system, N, i, j) = tex2D(readTexture, i, j) - multiplier * tex2D(readTexture, k, j);
			}
		}
	}

}