#ifndef NG_CUDA_TRANSPOSE_2D_H
#define NG_CUDA_TRANSPOSE_2D_H

#include <cuda.h>

void ng_cuda_transpose_zeropad_c2c_2d(
	cudaStream_t s0, cudaStream_t s1,
	int dim_x, int dim_y, // input size
	int exp_y, // exp_y >= dim_y
	const float * in, // size: dim_x * dim_y
	      float *out  // size: exp_y * dim_x (zero-padded in x-direction from dim_y -> exp_y)
);

void ng_cuda_transpose_unpad_c2c_2d(
	cudaStream_t s0,
	int dim_x, int dim_y, // input size
	int red_x, // red_x <= dim_x
	const float * in, // size: dim_x * dim_y
	      float *out  // size: dim_y * red_x
);

#endif
