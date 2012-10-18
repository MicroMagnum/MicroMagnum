#ifndef TRANSPOSE_UNPAD_H
#define TRANSPOSE_UNPAD_H

// XYZ -> ZxY
void cuda_transpose_unpad_c2c(
	int dim_x, int dim_y, int dim_z, // input size
	int red_x, // red_x <= dim_x
	const float *in_x, const float *in_y, const float *in_z,  // size: dim_x * dim_y * dim_z
	      float *out_x,     float *out_y,      float *out_z); // size: dim_z * red_x * dim_y (cut in x-direction [of input] from dim_x->red_x)

// XYZ -> ZxY, scalar version
void cuda_transpose_unpad_c2c(
	int dim_x, int dim_y, int dim_z, // input size
	int red_x, // red_x <= dim_x
	const float *in,  // size: dim_x * dim_y * dim_z
	      float *out); // size: dim_z * red_x * dim_y (cut in x-direction [of input] from dim_x->red_x)

#endif
