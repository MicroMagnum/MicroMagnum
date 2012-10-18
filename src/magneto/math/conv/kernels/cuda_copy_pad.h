#ifndef CUDA_COPY_PAD_H
#define CUDA_COPY_PAD_H

/*
 *  Input size: dim_x * dim_y * dim_z
 * Output size: exp_x * exp_y * exp_z
 */
void cuda_copy_pad_r2r(
	int dim_x, int dim_y, int dim_z,
	int exp_x,
	const float * in_x, const float * in_y, const float * in_z,
	      float *out_x,       float *out_y,       float *out_z
);

// scalar version
void cuda_copy_pad_r2r(
	int dim_x, int dim_y, int dim_z,
	int exp_x,
	const float * in_x,
	      float *out_x
);

/// double->float versions ///

void cuda_copy_pad_r2r(
	int dim_x, int dim_y, int dim_z,
	int exp_x,
	const double * in_x, const double * in_y, const double * in_z,
	       float *out_x,        float *out_y,        float *out_z
);

void cuda_copy_pad_r2r(
	int dim_x, int dim_y, int dim_z,
	int exp_x,
	const double * in_x,
	       float *out_x);

#endif
