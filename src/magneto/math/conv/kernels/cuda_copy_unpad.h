#ifndef CUDA_COPY_UNPAD_H
#define CUDA_COPY_UNPAD_H

void cuda_copy_unpad_r2r(
	int dim_x, int dim_y, int dim_z,
	int red_x,
	const float * in_x, const float * in_y, const float * in_z,
	      float *out_x,       float *out_y,       float *out_z
);

void cuda_copy_unpad_r2r(
	int dim_x, int dim_y, int dim_z,
	int red_x,
	const float * in_x,
	      float *out_x
);

/// float->double versions ///

void cuda_copy_unpad_r2r(
	int dim_x, int dim_y, int dim_z,
	int red_x,
	const  float * in_x,  const float * in_y,  const float * in_z,
	      double *out_x,       double *out_y,       double *out_z
);

void cuda_copy_unpad_r2r(
	int dim_x, int dim_y, int dim_z,
	int red_x,
	const  float * in_x,
	      double *out_x
);

#endif
