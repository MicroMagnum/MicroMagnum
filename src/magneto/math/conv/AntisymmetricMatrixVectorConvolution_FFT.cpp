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

#include "AntisymmetricMatrixVectorConvolution_FFT.h"

#include "TensorFieldSetup.h"

#include "kernels/cpu_multiplication.h"
#ifdef HAVE_CUDA
#include "kernels/cuda_multiplication.h"
#endif

AntisymmetricMatrixVectorConvolution_FFT::AntisymmetricMatrixVectorConvolution_FFT(const Matrix &lhs, int dim_x, int dim_y, int dim_z)
	: MatrixVectorConvolution_FFT(dim_x, dim_y, dim_z, lhs.getShape().getDim(1), lhs.getShape().getDim(2), lhs.getShape().getDim(3))
{
	assert(lhs.getShape().getDim(0) == 3);
	
	// Allocate buffers
	for (int e=0; e<3; ++e) {
		if (!is_2d) {
			N.re[e] = Matrix(Shape(exp_z, exp_x/2+1, exp_y));
			N.im[e] = Matrix(Shape(exp_z, exp_x/2+1, exp_y));
		} else {
			N.re[e] = Matrix(Shape(exp_y, exp_z, exp_x/2+1));
			N.im[e] = Matrix(Shape(exp_y, exp_z, exp_x/2+1));
		}
	}

	// Setup tensor field
	TensorFieldSetup setup(3, dim_x, dim_y, dim_z, exp_x, exp_y, exp_z);
	ComplexMatrix lhs2 = setup.transformTensorField(lhs);
	Matrix *foo1[3] = {&N.re[0], &N.re[1], &N.re[2]};
	Matrix *foo2[3] = {&N.im[0], &N.im[1], &N.im[2]};
	if (!is_2d) {
		setup.unpackTransformedTensorField_xyz_to_zxy(lhs2, foo1, foo2);
	} else {
		setup.unpackTransformedTensorField_xyz_to_yzx(lhs2, foo1, foo2);
	}
}

AntisymmetricMatrixVectorConvolution_FFT::~AntisymmetricMatrixVectorConvolution_FFT()
{
}

void AntisymmetricMatrixVectorConvolution_FFT::calculate_multiplication(double *inout_x, double *inout_y, double *inout_z)
{
	Matrix::ro_accessor N_re_acc[3] = {N.re[0], N.re[1], N.re[2]};
	Matrix::ro_accessor N_im_acc[3] = {N.im[0], N.im[1], N.im[2]};

	cpu_multiplication_antisymmetric(
		(exp_x/2+1) * exp_y * exp_z, // num_elements
		N_re_acc[0].ptr(), N_re_acc[1].ptr(), N_re_acc[2].ptr(),
		N_im_acc[0].ptr(), N_im_acc[1].ptr(), N_im_acc[2].ptr(),
		inout_x, inout_y, inout_z
	);
}

#ifdef HAVE_CUDA
void AntisymmetricMatrixVectorConvolution_FFT::calculate_multiplication_cuda(float *inout_x, float *inout_y, float *inout_z)
{
	Matrix::const_cu32_accessor N_re_acc[3] = {N.re[0], N.re[1], N.re[2]};
	Matrix::const_cu32_accessor N_im_acc[3] = {N.im[0], N.im[1], N.im[2]};

	cuda_multiplication_antisymmetric(
		(exp_x/2+1) * exp_y * exp_z, // num_elements
		N_re_acc[0].ptr(), N_re_acc[1].ptr(), N_re_acc[2].ptr(),
		N_im_acc[0].ptr(), N_im_acc[1].ptr(), N_im_acc[2].ptr(),
		inout_x, inout_y, inout_z
	);
}
#endif
