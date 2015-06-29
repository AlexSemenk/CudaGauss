#include "cuda_def.h"
#include <stdlib.h>
#include <stdio.h>
#include "matrix.h"

namespace global_gauss3 {

	#define BLOCK_X_SIZE 8		// Q(2, 2)
	#define BLOCK_Y_SIZE 8		// Q(3, 2)
//	#define BLOCK_Z_SIZE 1 		// Q(1, 2)

	#define Q22 BLOCK_X_SIZE
	#define Q32 BLOCK_Y_SIZE
//	#define Q12 BLOCK_Z_SIZE

	#define THREAD_X_SIZE 8		// r(2, 2)
	#define THREAD_Y_SIZE 8		// r(3, 2)
	#define THREAD_Z_SIZE 64	// r(1, 2)

	#define R22 THREAD_X_SIZE
	#define R32 THREAD_Y_SIZE
	#define R12 THREAD_Z_SIZE

	/*
	#TODO = SHARED MEMORY
	*/


	/*
	  ____________________________________________
	 |                                            |
	 |                                            |
	 |           _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _|
	 |          :_|____________const______________|
	 |          :c|_ _:_ _|       |       |       |
	 |          :o|___:___|_______|_______|_______|
	 |          :n|       |       |       |       |
	 |		    :s|_______|_______|_______|_______|
	 |          :t|       |       |       |       |
	 |__________:_|_______|_______|_______|_______|

	*/

	__global__ void solve_kernel(DeviceSystem<float> system, int k_gl, int t);
	__global__ void clean_kernel(DeviceSystem<float> system);
	void solve(DeviceSystem<float>& system);



	extern "C" void global_gauss_3d3(System<float>& system) {
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
		const int N = system.dim;
		const int Q1 = div_ceiling(N-1, THREAD_Z_SIZE);
		// solve
		for (int k_gl=0; k_gl<Q1; k_gl++) {
			const int Q2 = div_ceiling(N-1 - k_gl*THREAD_Z_SIZE, BLOCK_X_SIZE*THREAD_X_SIZE);
			const int Q3 = div_ceiling(N - k_gl*THREAD_Z_SIZE, BLOCK_Y_SIZE*THREAD_Y_SIZE);
			dim3 blocks(Q2, Q3);
			dim3 threads(BLOCK_X_SIZE, BLOCK_Y_SIZE);
			for(int t=0; t<Q2+Q3; t++) {
				//size_t sharedMemSize = sizeof(float) * BLOCK_X_SIZE * THREAD_X_SIZE * BLOCK_Y_SIZE * THREAD_Y_SIZE;
				solve_kernel<<<blocks, threads>>>(system, k_gl, t);
			}
		}
		// clean
		dim3 blocks(div_ceiling(N, 8), div_ceiling(N, 8));
		dim3 threads(8, 8);
		clean_kernel<<<blocks, threads>>>(system);
	}

	/* 

	     ri - relatiove i position
		 di - i position deviation

		 di = 0                       di = 1                       di = 4
		 dj = 0                       dj = 0                       dj = 1
		 ______________               ______________               ______________
		|____|####|____|  ri=0 rj=1  |____|____|____|             |____|____|____|
		|####|____|____|  ri=1 rj=0  |____|____|####|  ri=0 rj=2  |____|____|____|
		|____|____|____|             |____|####|____|  ri=1 rj=1  |____|____|____|
		|____|____|____|             |####|____|____|  ri=2 rj=0  |____|____|____|
		|____|____|____|             |____|____|____|             |____|____|####|  ri=0 rj=1
		|____|____|____|             |____|____|____|             |____|####|____|  ri=1 rj=0

		 _______________
		|	|   | b |   |
		|___|___|___|___|
		|___|___|___|___|
		|_c_|___|_a_|___|
		|___|___|___|___|
		|___|___|___|___|

	*/

	__global__ void solve_kernel(DeviceSystem<float> s, int k_gl, int t) {
		int Q2 = gridDim.x;
		int Q3 = gridDim.y;
		
		int N = s.dim;
		int start_k = k_gl*THREAD_Z_SIZE;

		int i_block = blockIdx.x;
		int j_block = blockIdx.y;
	
		if (i_block + j_block != t) {
			return;
		}

		const int block_i_gl = i_block * BLOCK_X_SIZE * THREAD_X_SIZE + (start_k + 1);
		const int block_j_gl = j_block * BLOCK_Y_SIZE * THREAD_Y_SIZE + (start_k + 1);

		const int i_gl = (i_block * BLOCK_X_SIZE + threadIdx.x) * THREAD_X_SIZE + (start_k + 1);
		const int j_gl = (j_block * BLOCK_Y_SIZE + threadIdx.y) * THREAD_Y_SIZE + (start_k + 1);

		const int first_ai = threadIdx.x * THREAD_X_SIZE;
		const int first_aj = threadIdx.y * THREAD_Y_SIZE;
		const int aSizeX = BLOCK_X_SIZE*THREAD_X_SIZE;
		const int aSizeY = BLOCK_Y_SIZE*THREAD_Y_SIZE;
		__shared__ float a[aSizeX][aSizeY];	// main block for read, write.
		for (int ai = first_ai, si = i_gl; si < min(i_gl + THREAD_X_SIZE, N); ai++, si++) {
			for(int aj = first_aj, sj = j_gl; sj < min(j_gl + THREAD_Y_SIZE, N+1); aj++, sj++) {
				a[ai][aj] = s[si][sj];
			}
		}

		for(int ks = start_k, kia = ks - block_i_gl, kja = ks - block_j_gl; ks < min(start_k+THREAD_Z_SIZE, N-1); ks++, kia++, kja++) {
			float skk = kja >= 0 && kja < aSizeX && kia >= 0 && kia < aSizeY ? a[kia][kja] : s[ks][ks];
			for (int is = max(i_gl, ks+1), ia = is - block_i_gl; is < min(i_gl + THREAD_X_SIZE, N); is++, ia++) {
				float sik = kja >= 0 && kja < aSizeX ? a[ia][kja] : s[is][ks];
				float l = sik / skk;
				for(int js = max(j_gl, ks+1), ja = js - block_j_gl; js < min(j_gl + THREAD_Y_SIZE, N+1); js++, ja++) {
					float skj = ks < block_i_gl ? s[ks][js] : a[kia][ja];
					a[ia][ja] = a[ia][ja] - l * skj;
				}
			}
			__syncthreads();
		}

		for (int ai = first_ai, si = i_gl; si < min(i_gl + THREAD_X_SIZE, N); ai++, si++) {
			for(int aj = first_aj, sj = j_gl; sj < min(j_gl + THREAD_Y_SIZE, N+1); aj++, sj++) {
				s[si][sj] = a[ai][aj];
			}
		}

	}

	__global__ void clean_kernel(DeviceSystem<float> a) {
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		int N = a.dim;
		if (i<N && j<N && i>j) {
			a[i][j] = 0;
		}
	}

}
