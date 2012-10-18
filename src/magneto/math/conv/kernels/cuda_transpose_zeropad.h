#ifndef TRANSPOSE_ZEROPAD_H
#define TRANSPOSE_ZEROPAD_H

// xyz -> Yzx
void cuda_transpose_zeropad_c2c(
	int dim_x, int dim_y, int dim_z, // input size
	int exp_y,
	const float *in_x, const float *in_y, const float *in_z,
	      float *out_x, float *out_y, float *out_z);

#endif
