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

#include "SymmetricMatrixVectorConvolution_FFT.h"

#include "TensorFieldSetup.h"

#include "kernels/cpu_multiplication.h"
#ifdef HAVE_CUDA
#include "kernels/cuda_multiplication.h"
#endif

#include "Logger.h"

SymmetricMatrixVectorConvolution_FFT::SymmetricMatrixVectorConvolution_FFT(const Matrix &lhs, int dim_x, int dim_y, int dim_z)
	: MatrixVectorConvolution_FFT(dim_x, dim_y, dim_z, lhs.getShape().getDim(1), lhs.getShape().getDim(2), lhs.getShape().getDim(3))
{
	assert(lhs.getShape().getDim(0) == 6);

	// Allocate buffers
	for (int e=0; e<6; ++e) {
		if (!is_2d) {
			N.re[e] = Matrix(Shape(exp_z, exp_x/2+1, exp_y));
			N.im[e] = Matrix(Shape(exp_z, exp_x/2+1, exp_y));
		} else {
			N.re[e] = Matrix(Shape(exp_y, exp_z, exp_x/2+1));
			N.im[e] = Matrix(Shape(exp_y, exp_z, exp_x/2+1));
		}
	}

	// Setup tensor field
	TensorFieldSetup setup(6, dim_x, dim_y, dim_z, exp_x, exp_y, exp_z);
	ComplexMatrix lhs2 = setup.transformTensorField(lhs);
	Matrix *foo1[6] = {&N.re[0], &N.re[1], &N.re[2], &N.re[3], &N.re[4], &N.re[5]};
	Matrix *foo2[6] = {&N.im[0], &N.im[1], &N.im[2], &N.im[3], &N.im[4], &N.im[5]};
	if (!is_2d) {
		LOG_DEBUG << "Convolution is 3D";
		setup.unpackTransformedTensorField_xyz_to_zxy(lhs2, foo1, foo2);
	} else {
		LOG_DEBUG << "Convolution is 2D";
		setup.unpackTransformedTensorField_xyz_to_yzx(lhs2, foo1, foo2);
	}

	// Optimization: Handle cases where imaginary parts of FFT(N[i]) (=N.im[i]) are zero.
	/*for (int i=0; i<6; ++i) {
		cout << i << " absmax(re): " << N.re[i].absMax() << endl;
		cout << i << " absmax(im): " << N.im[i].absMax() << endl;
	}*/
}

SymmetricMatrixVectorConvolution_FFT::~SymmetricMatrixVectorConvolution_FFT()
{
}

void SymmetricMatrixVectorConvolution_FFT::calculate_multiplication(double *inout_x, double *inout_y, double *inout_z)
{
	Matrix::ro_accessor N_re_acc[6] = {N.re[0], N.re[1], N.re[2], N.re[3], N.re[4], N.re[5]};
	Matrix::ro_accessor N_im_acc[6] = {N.im[0], N.im[1], N.im[2], N.im[3], N.im[4], N.im[5]};
	cpu_multiplication_symmetric(
		(exp_x/2+1) * exp_y * exp_z, // num_elements
		N_re_acc[0].ptr(), N_re_acc[1].ptr(), N_re_acc[2].ptr(), N_re_acc[3].ptr(), N_re_acc[4].ptr(), N_re_acc[5].ptr(),
		N_im_acc[0].ptr(), N_im_acc[1].ptr(), N_im_acc[2].ptr(), N_im_acc[3].ptr(), N_im_acc[4].ptr(), N_im_acc[5].ptr(),
		inout_x, inout_y, inout_z
	);
}

#ifdef HAVE_CUDA
void SymmetricMatrixVectorConvolution_FFT::calculate_multiplication_cuda(float *inout_x, float *inout_y, float *inout_z)
{
	Matrix::const_cu32_accessor N_re_acc[6] = {N.re[0], N.re[1], N.re[2], N.re[3], N.re[4], N.re[5]};
	Matrix::const_cu32_accessor N_im_acc[6] = {N.im[0], N.im[1], N.im[2], N.im[3], N.im[4], N.im[5]};
	cuda_multiplication_symmetric(
		(exp_x/2+1) * exp_y * exp_z, // num_elements
		N_re_acc[0].ptr(), N_re_acc[1].ptr(), N_re_acc[2].ptr(), N_re_acc[3].ptr(), N_re_acc[4].ptr(), N_re_acc[5].ptr(),
		N_im_acc[0].ptr(), N_im_acc[1].ptr(), N_im_acc[2].ptr(), N_im_acc[3].ptr(), N_im_acc[4].ptr(), N_im_acc[5].ptr(),
		inout_x, inout_y, inout_z
	);
}
#endif

