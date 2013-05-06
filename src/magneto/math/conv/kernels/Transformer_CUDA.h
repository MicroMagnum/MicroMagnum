/*
 * Copyright 2012, 2013 by the Micromagnum authors.
 *
 * This file is part of MicroMagnum.
 * 
 * MicroMagnum is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * MicroMagnum is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with MicroMagnum.  If not, see <http://www.gnu.org/licenses/>.
 */

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
