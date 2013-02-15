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

#ifndef VECTOR_VECTOR_CONVOLUTION_FFT_H
#define VECTOR_VECTOR_CONVOLUTION_FFT_H

#include <memory>

#include "matrix/matty.h"

#include "kernels/Transposer_CPU.h"
#include "kernels/Transformer_CPU.h"
#include <fftw3.h>
#ifdef HAVE_CUDA
#include "kernels/Transposer_CUDA.h"
#include "kernels/Transformer_CUDA.h"
#include <cufft.h>
#endif

//class VectorVectorConvolutionGradient_FFT //TODO: Rename to this name (?)
class VectorVectorConvolution_FFT
{
public:
	VectorVectorConvolution_FFT(const Matrix &lhs, int dim_x, int dim_y, int dim_z, double delta_x, double delta_y, double delta_z);
	virtual ~VectorVectorConvolution_FFT();

	virtual void execute(const VectorMatrix &rhs, VectorMatrix &res);

private:
	bool use_cuda;

	// Problem size
	int dim_x, dim_y, dim_z;
	int exp_x, exp_y, exp_z;
	double delta_x, delta_y, delta_z;

	// Buffers
	struct scratch_buf {
		Matrix M[3];
	} s1, s2;

	struct tensor_buf {
		Matrix re[3], im[3];
	} S;

	void setupInverseTransforms();

	// Transpose and transform algorithms
	std::auto_ptr<Transposer_CPU> transposer;
	std::auto_ptr<Transformer_CPU> transformer;
	fftw_plan plan_x_c2r, plan_y_inv; // we need different X,Y back transforms than in Transposer_CPU...
	void calculate_multiplication(double *inout_x, double *in_y, double *in_z);
	
#ifdef HAVE_CUDA
	std::auto_ptr<Transposer_CUDA> transposer_cuda;
	std::auto_ptr<Transformer_CUDA> transformer_cuda;
	cufftHandle plan_x_c2r_cuda, plan_y_inv_cuda;
#endif
};

#endif
