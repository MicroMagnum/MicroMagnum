#ifndef CPU_TRANSPOSE_ZEROPAD_3D_H
#define CPU_TRANSPOSE_ZEROPAD_3D_H

void cpu_transpose_zeropad_c2c(
	int dim_x, int dim_y, int dim_z, // input size
	int exp_y, // exp_x >= dim_x
	const double  *in_x, const double  *in_y, const double  *in_z,   // size: dim_x * dim_y * dim_z
	      double *out_x,       double *out_y,       double *out_z);  // size: exp_y * dim_z * dim_x

#endif
