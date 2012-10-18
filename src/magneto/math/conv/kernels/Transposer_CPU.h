#ifndef TRANSPOSER_CPU_H
#define TRANSPOSER_CPU_H

#include "matrix/matty.h"

#include <fftw3.h>

class Transposer_CPU
{
public:
	Transposer_CPU(int dim_x, int dim_y, int dim_z, int exp_x, int exp_y, int exp_z);
	~Transposer_CPU();

	void copy_pad(const VectorMatrix &M, double *out_x, double *out_y, double *out_z);
	void transpose_zeropad_yzx(const double *in_x, const double *in_y, const double *in_z, double *out_x, double *out_y, double *out_z);
	void transpose_zeropad_zxy(const double *in_x, const double *in_y, const double *in_z, double *out_x, double *out_y, double *out_z);
	void transpose_unpad_yzx(const double *in_x, const double *in_y, const double *in_z, double *out_x, double *out_y, double *out_z);
	void transpose_unpad_xyz(const double *in_x, const double *in_y, const double *in_z, double *out_x, double *out_y, double *out_z);
	void copy_unpad(const double *in_x, const double *in_y, const double *in_z, VectorMatrix &H);

private:
	const int dim_x, dim_y, dim_z;
	const int exp_x, exp_y, exp_z;

	fftw_plan plan_unpad_zxy_yzx, plan_unpad_yzx_xyz;

	void initPlans();
	void deinitPlans();
};

#endif
