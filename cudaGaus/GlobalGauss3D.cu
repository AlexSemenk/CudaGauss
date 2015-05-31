#include "cuda_def.h"
#include <stdlib.h>
#include <stdio.h>
#include "matrix.h"

namespace global_gauss {

	#define BLOCK_X_SIZE 8		// r(2)
	#define BLOCK_Y_SIZE 8		// r(3)
	#define BLOCK_Z_SIZE 64		// r(1)

	#define R1 BLOCK_Z_SIZE
	#define R2 BLOCK_X_SIZE
	#define R3 BLOCK_Y_SIZE

	#define THREAD_X_SIZE 2 // r(2, 2)
	#define THREAD_Y_SIZE 2 // r(3, 2)
//	#define THERAK_K_SIZE - // r(1, 2)

	#define R22 THREAD_X_SIZE
	#define R32 THREAD_Y_SIZE
//	#define R12 THERAK_K_SIZE

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

	extern "C" void global_gauss_3d(System<float>& system) {
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
		int Q1 = div_ceiling(N-1, R1);
		// solve
		for (int k_gl=0; k_gl<Q1; k_gl++) {
			int Q2 = div_ceiling(N-1 - k_gl*R1, R2*THREAD_X_SIZE);
			int Q3 = div_ceiling(N - k_gl*R1, R3*THREAD_Y_SIZE);
			for (int t=1; t < Q2+Q3; t++) {
				int block_num;
				if (t < min(Q2, Q3)) {
					block_num = t;
				} else if (t > max(Q2, Q3)) {
					block_num = (Q2 + Q3) - t;
				} else {
					block_num = min(Q2, Q3);
				}
				dim3 blocks(block_num);
				dim3 threads(BLOCK_X_SIZE, BLOCK_Y_SIZE);
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

	*/

	__global__ void solve_kernel(DeviceSystem<float> s, int k_gl, int t) {

		int N = s.dim;
		int start_k = k_gl*R1;
		int Q2 = dev_div_ceiling(N-1 - start_k, R2*THREAD_X_SIZE);
		int Q3 = dev_div_ceiling(N - start_k, R3*THREAD_Y_SIZE);
		
		int di_block = max(0, t - Q3);
		int dj_block = max(0, t - Q2);
		int ri_block = blockIdx.x;
		int rj_block = gridDim.x - 1 - blockIdx.x;
		int i_block = ri_block + di_block;
		int j_block = rj_block + dj_block;
	
		const int i_gl = (i_block * BLOCK_X_SIZE + threadIdx.x) * THREAD_X_SIZE + (start_k + 1);
		const int j_gl = (j_block * BLOCK_Y_SIZE + threadIdx.y) * THREAD_Y_SIZE + (start_k + 1);

		const int block_i_gl = i_block * BLOCK_X_SIZE * THREAD_X_SIZE + (start_k + 1);
		const int block_j_gl = j_block * BLOCK_Y_SIZE * THREAD_Y_SIZE + (start_k + 1);

		const int first_ai = threadIdx.x * THREAD_X_SIZE;
		const int first_aj = threadIdx.y * THREAD_Y_SIZE;

		const int sharedWidth = BLOCK_X_SIZE*THREAD_X_SIZE;
		const int sharedHeight = BLOCK_Y_SIZE*THREAD_Y_SIZE;
		__shared__ float a[sharedHeight][sharedWidth];
		for (int ai = first_ai, si = i_gl; si < min(i_gl + THREAD_X_SIZE, N); ai++, si++) {
			for(int aj = first_aj, sj = j_gl; sj < min(j_gl + THREAD_Y_SIZE, N+1); aj++, sj++) {
				a[ai][aj] = s[si][sj];
			}
		}

		for(int ks = start_k, kia = ks - block_i_gl, kja = ks - block_j_gl; ks < min(start_k+R1, N-1); ks++, kia++, kja++) {
			for (int is = max(i_gl, ks+1), ia = is - block_i_gl; is < min(i_gl + THREAD_X_SIZE, N); is++, ia++) {
				float aik = kja >= 0 && kja < sharedWidth ? a[ia][kja] : s[is][ks];
				float akk = kja >= 0 && kja < sharedWidth && kia >= 0 && kia < sharedHeight ? a[kia][kja] : s[ks][ks];
				float l = aik / akk;
				for(int js = max(j_gl, ks+1), ja = js - block_j_gl; js < min(j_gl + THREAD_Y_SIZE, N+1); js++, ja++) {
					float akj = ks < block_i_gl ? s[ks][js] : a[kia][ja];
					a[ia][ja] = a[ia][ja] - l * akj;
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