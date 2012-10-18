#ifndef TRANSPOSE_ZEROPAD_2D_H
#define TRANSPOSE_ZEROPAD_2D_H

// xyz -> Yzx
void cuda_transpose_zeropad_c2c_2d(
	int dim_x, int dim_y, // input size
	int exp_y, // exp_y >= dim_y
	const float *in_x, const float *in_y, const float *in_z, // size: dim_x * dim_y
	      float *out_x, float *out_y, float *out_z);         // size: exp_y * dim_x (zero-padded in x-direction from dim_y -> exp_y)

#endif
