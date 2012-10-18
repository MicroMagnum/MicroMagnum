#ifndef NG_TRANSPOSE_ZEROPAD_H
#define NG_TRANSPOSE_ZEROPAD_H

#include <cuda_runtime_api.h>

// xyz -> Yzx
void ng_cuda_transpose_zeropad_c2c(
	cudaStream_t s0, cudaStream_t s1,
	int dim_x, int dim_y, int dim_z, // input size
	int exp_y, // exp_y >= dim_y
	const float *in, // size: dim_x * dim_y * dim_z
	      float *out // size: exp_y * dim_z * dim_x
);

#endif
