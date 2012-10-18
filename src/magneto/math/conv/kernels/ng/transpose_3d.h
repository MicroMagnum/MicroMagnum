#ifndef NG_CUDA_TRANSPOSE_3D_H
#define NG_CUDA_TRANSPOSE_3D_H

#include <cuda.h>

void ng_cuda_transpose_zeropad_c2c_3d( // xyz -> Yzx
	cudaStream_t s0, cudaStream_t s1,
	int dim_x, int dim_y, int dim_z, // input size
	int exp_y, // exp_y >= dim_y
	const float * in, // size: dim_x * dim_y * dim_z
	      float *out  // size: exp_y * dim_z * dim_x
);

void ng_cuda_transpose_unpad_c2c_3d( // XYZ -> ZxY
	int dim_x, int dim_y, int dim_z, // input size
	int red_x, // red_x <= dim_x
	const float *in,   // size: dim_x * dim_y * dim_z
	      float *out   // size: dim_z * red_x * dim_y (cut in x-direction from dim_x->red_x)
);

#endif
