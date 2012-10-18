#ifndef TRANSFORMER_CUDA_H
#define TRANSFORMER_CUDA_H

#include <cufft.h>

class Transformer_CUDA
{
public:
	Transformer_CUDA(int dim_x, int dim_y, int dim_z, int exp_x, int exp_y, int exp_z);
	~Transformer_CUDA();

	void transform_forward_x(const float *in, float *out);
	void transform_forward_y(float *inout);
	void transform_forward_z(float *inout);
	void transform_inverse_z(float *inout);
	void transform_inverse_y(float *inout);
	void transform_inverse_x(const float *in, float *out);

private:
	const int dim_x, dim_y, dim_z;
	const int exp_x, exp_y, exp_z;

	// CUFFT plan handles
	cufftHandle plan_x_r2c, plan_x_c2r;
	cufftHandle plan_y_c2c;
	cufftHandle plan_z_c2c;
};

#endif
