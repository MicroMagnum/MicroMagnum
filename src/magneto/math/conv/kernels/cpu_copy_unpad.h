#ifndef CPU_COPY_UNPAD_H
#define CPU_COPY_UNPAD_H

void cpu_copy_unpad_c2r(
	int dim_x, int dim_y, int dim_z,
	int red_x,
	const double  *in_x, const double  *in_y, const double  *in_z,
	      double *out_x,       double *out_y,       double *out_z);

void cpu_copy_unpad_r2r(
	int dim_x, int dim_y, int dim_z,
	int red_x,
	const double * in_x, const double * in_y, const double * in_z,
	      double *out_x,       double *out_y,       double *out_z
);

// scalar version...
void cpu_copy_unpad_c2r(
	int dim_x, int dim_y, int dim_z,
	int red_x,
	const double  *in_x,
	      double *out_x);

void cpu_copy_unpad_r2r(
	int dim_x, int dim_y, int dim_z,
	int red_x,
	const double * in_x,
	      double *out_x);

#endif
