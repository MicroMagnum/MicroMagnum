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

#ifndef SYMMETRIC_MATRIX_VECTOR_CONVOLUTION_FFT_H
#define SYMMETRIC_MATRIX_VECTOR_CONVOLUTION_FFT_H

#include "MatrixVectorConvolution_FFT.h"

class SymmetricMatrixVectorConvolution_FFT : public MatrixVectorConvolution_FFT
{
public:
	SymmetricMatrixVectorConvolution_FFT(const Matrix &lhs, int dim_x, int dim_y, int dim_z);
	virtual ~SymmetricMatrixVectorConvolution_FFT();

private:
	virtual void calculate_multiplication(double *inout_x, double *inout_y, double *inout_z);
#ifdef HAVE_CUDA
	virtual void calculate_multiplication_cuda(float *inout_x, float *inout_y, float *inout_z);
#endif

	// Buffers
	struct tensor_buf {
		Matrix re[6], im[6];
	} N;
};

#endif
