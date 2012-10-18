#ifndef TRANSPOSE_ZEROPAD_3D_H
#define TRANSPOSE_ZEROPAD_3D_H

// xyz -> Yzx
void cuda_transpose_zeropad_c2c_3d(
	int dim_x, int dim_y, int dim_z, // input size
	int exp_y, // exp_y >= dim_y
	const float *in_x, const float *in_y, const float *in_z, // size: dim_x * dim_y
	      float *out_x, float *out_y, float *out_z);         // size: dim_y * red_x

#endif
