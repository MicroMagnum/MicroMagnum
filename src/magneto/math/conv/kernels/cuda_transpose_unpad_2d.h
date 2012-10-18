#ifndef TRANSPOSE_UNPAD_2D_H
#define TRANSPOSE_UNPAD_2D_H

// XY -> xY
void cuda_transpose_unpad_c2c_2d(
	int dim_x, int dim_y, // input size
	int red_x, // red_x <= dim_x
	const float *in_x, const float *in_y, const float *in_z, // size: dim_x * dim_y
	      float *out_x, float *out_y, float *out_z);         // size: dim_y * red_x

// XY -> xY, scalar version
void cuda_transpose_unpad_c2c_2d(
	int dim_x, int dim_y, // input size
	int red_x, // red_x <= dim_x
	const float *in,   // size: dim_x * dim_y
	      float *out); // size: dim_y * red_x

#endif
