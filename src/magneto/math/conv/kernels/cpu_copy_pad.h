#ifndef CPU_COPY_PAD_H
#define CPU_COPY_PAD_H

/*
 *  Input size: dim_x * dim_y * dim_z
 * Output size: exp_x * exp_y * exp_z
 */
void cpu_copy_pad_r2c(
	int dim_x, int dim_y, int dim_z,
	int exp_x,
	const double * in_x, const double * in_y, const double * in_z, 
	      double *out_x,       double *out_y,       double *out_z
);

/*
 *  Input size: dim_x * dim_y * dim_z
 * Output size: exp_x * exp_y * exp_z
 */
void cpu_copy_pad_r2r(
	int dim_x, int dim_y, int dim_z,
	int exp_x,
	const double * in_x, const double * in_y, const double * in_z,
	      double *out_x,       double *out_y,       double *out_z
);

#endif
