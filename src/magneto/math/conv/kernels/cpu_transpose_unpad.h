#ifndef CPU_TRANSPOSE_UNPAD_3D_H
#define CPU_TRANSPOSE_UNPAD_3D_H

void cpu_transpose_unpad_c2c(
	int dim_x, int dim_y, int dim_z, // input size
	int red_x, // red_x <= dim_x
	const double  *in_x, const double  *in_y, const double  *in_z,  // size: dim_x * dim_y * dim_z
	      double *out_x,      double *out_y,       double *out_z);  // size: dim_z * red_x * dim_y (cut in x-direction [of input] from dim_x->red_x)

// scalar version...
void cpu_transpose_unpad_c2c(
	int dim_x, int dim_y, int dim_z, // input size
	int red_x, // red_x <= dim_x
	const double  *in_x,  // size: dim_x * dim_y * dim_z
	      double *out_x);  // size: dim_z * red_x * dim_y (cut in x-direction [of input] from dim_x->red_x)

#endif

