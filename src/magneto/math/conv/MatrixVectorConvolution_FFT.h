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

#ifndef MATRIX_VECTOR_CONVOLUTION_FFT_H
#define MATRIX_VECTOR_CONVOLUTION_FFT_H

#include <memory>

#include "matrix/matty.h"

#include "kernels/Transposer_CPU.h"
#include "kernels/Transformer_CPU.h"
#ifdef HAVE_CUDA
#include "kernels/Transposer_CUDA.h"
#include "kernels/Transformer_CUDA.h"
#endif

class MatrixVectorConvolution_FFT
{
public:
	MatrixVectorConvolution_FFT(int dim_x, int dim_y, int dim_z, int exp_x, int exp_y, int exp_z);
	virtual ~MatrixVectorConvolution_FFT();

	virtual void execute(const VectorMatrix &rhs, VectorMatrix &res);

protected:
	virtual void calculate_multiplication(double *inout_x, double *inout_y, double *inout_z) = 0;
#ifdef HAVE_CUDA
	virtual void calculate_multiplication_cuda(float *inout_x, float *inout_y, float *inout_z) = 0;
#endif

	bool use_cuda;
	bool is_2d;

	// Problem size
	int dim_x, dim_y, dim_z;
	int exp_x, exp_y, exp_z;

	// Buffers
	struct scratch_buf {
		Matrix M[3];
	} s1, s2;

	// Transpose and transform algorithms
	std::auto_ptr<Transposer_CPU> transposer;
	std::auto_ptr<Transformer_CPU> transformer;
#ifdef HAVE_CUDA
	std::auto_ptr<Transposer_CUDA> transposer_cuda;
	std::auto_ptr<Transformer_CUDA> transformer_cuda;
#endif
};

#endif
