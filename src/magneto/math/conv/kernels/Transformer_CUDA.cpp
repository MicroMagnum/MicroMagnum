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

#include "Transformer_CUDA.h"

#include "matrix/device/cuda_tools.h"

#include <stdexcept>
#include <iostream>
using namespace std;

Transformer_CUDA::Transformer_CUDA(int dim_x, int dim_y, int dim_z, int exp_x, int exp_y, int exp_z)
	: dim_x(dim_x), dim_y(dim_y), dim_z(dim_z), exp_x(exp_x), exp_y(exp_y), exp_z(exp_z)
{
	// Create CUFFT plans
	cufftResult res;
	
	// a) Plan for first FFT in x-direction / last IFFT in x-direction
	res = cufftPlan1d(&plan_x_r2c, exp_x /*transform size*/, CUFFT_R2C, dim_y*dim_z /*batch size*/);
	if (res != CUFFT_SUCCESS) throw std::runtime_error("Error initializing cufft plan (1)");

	res = cufftPlan1d(&plan_x_c2r, exp_x /*transform size*/, CUFFT_C2R, dim_y*dim_z /*batch size*/);
	if (res != CUFFT_SUCCESS) throw std::runtime_error("Error initializing cufft plan (2)");

	// b) Plan for FFTs in y-direction
	res = cufftPlan1d(&plan_y_c2c, exp_y /*transf.size*/, CUFFT_C2C, (exp_x/2+1)*dim_z /*batch size*/);
	if (res != CUFFT_SUCCESS) throw std::runtime_error("Error initializing cufft plan (3)");

	// c) Plan for FFTs in z-direction
	res = cufftPlan1d(&plan_z_c2c, exp_z /*transf.size*/, CUFFT_C2C, (exp_x/2+1)*exp_y /*batch size*/);
	if (res != CUFFT_SUCCESS) throw std::runtime_error("Error initializing cufft plan (4)");
}

Transformer_CUDA::~Transformer_CUDA()
{
	cufftDestroy(plan_x_r2c);
	cufftDestroy(plan_x_c2r);
	cufftDestroy(plan_y_c2c);
	cufftDestroy(plan_z_c2c);
}

void Transformer_CUDA::transform_forward_y(float *inout)
{
	checkCufftSuccess(cufftExecC2C(plan_y_c2c, (cufftComplex*)inout, (cufftComplex*)inout, CUFFT_FORWARD));
}

void Transformer_CUDA::transform_forward_z(float *inout)
{
	checkCufftSuccess(cufftExecC2C(plan_z_c2c, (cufftComplex*)inout, (cufftComplex*)inout, CUFFT_FORWARD));
}

void Transformer_CUDA::transform_inverse_z(float *inout)
{
	checkCufftSuccess(cufftExecC2C(plan_z_c2c, (cufftComplex*)inout, (cufftComplex*)inout, CUFFT_INVERSE));
}

void Transformer_CUDA::transform_inverse_y(float *inout)
{
	checkCufftSuccess(cufftExecC2C(plan_y_c2c, (cufftComplex*)inout, (cufftComplex*)inout, CUFFT_INVERSE));
}

void Transformer_CUDA::transform_forward_x(const float *in, float *out)
{
	checkCufftSuccess(cufftExecR2C(plan_x_r2c, (cufftReal*)in, (cufftComplex*)out));
}

void Transformer_CUDA::transform_inverse_x(const float *in, float *out)
{
	checkCufftSuccess(cufftExecC2R(plan_x_c2r, (cufftComplex*)in, (cufftReal*)out));
}
