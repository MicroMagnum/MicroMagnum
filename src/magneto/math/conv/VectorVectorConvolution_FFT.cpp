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

#include "VectorVectorConvolution_FFT.h"

#include "TensorFieldSetup.h"
#include "Magneto.h"
#include "Logger.h"

#include "kernels/Transposer_CPU.h"
#include "kernels/Transformer_CPU.h"
#include "kernels/cpu_multiplication.h"
#include "kernels/cpu_transpose_unpad.h"
#include "kernels/cpu_copy_unpad.h"
#ifdef HAVE_CUDA
#include "kernels/Transposer_CUDA.h"
#include "kernels/Transformer_CUDA.h"
#include "kernels/cuda_multiplication.h"
#include "kernels/cuda_transpose_unpad.h"
#include "kernels/cuda_copy_unpad.h"
#endif

#include "math/gradient.h"

#include <stdexcept>
#include <iostream>
#include <fstream>
using namespace std;

static const int fftw_strategy = FFTW_MEASURE;
static const int R = 1;

VectorVectorConvolution_FFT::VectorVectorConvolution_FFT(const Matrix &lhs, int dim_x, int dim_y, int dim_z, double delta_x, double delta_y, double delta_z)
	: dim_x(dim_x), dim_y(dim_y), dim_z(dim_z), delta_x(delta_x), delta_y(delta_y), delta_z(delta_z)
{
	assert(lhs.getShape().getDim(0) == 3);
	exp_x = lhs.getShape().getDim(1);
	exp_y = lhs.getShape().getDim(2);
	exp_z = lhs.getShape().getDim(3);

	// Enable CUDA?
#ifdef HAVE_CUDA
	use_cuda = isCudaEnabled();
#else
	use_cuda = false;
#endif

	// Allocate buffers
	for (int e=0; e<3; ++e) {
		s1.M[e] = Matrix(Shape(2, exp_x/2+1, exp_y, exp_z));
		s2.M[e] = Matrix(Shape(2, exp_x/2+1, exp_y, exp_z));
	}

	for (int e=0; e<3; ++e) {
		S.re[e] = Matrix(Shape(exp_z, exp_x/2+1, exp_y));
		S.im[e] = Matrix(Shape(exp_z, exp_x/2+1, exp_y));
	}

	// Initializer transposer & transformer
	if (use_cuda) {
#ifdef HAVE_CUDA
		LOG_DEBUG << "Using CUDA routines for Vector-Vector convolution";
		transposer_cuda .reset(new  Transposer_CUDA(dim_x, dim_y, dim_z, exp_x, exp_y, exp_z));
		transformer_cuda.reset(new Transformer_CUDA(dim_x, dim_y, dim_z, exp_x, exp_y, exp_z));
		setupInverseTransforms();
#else
		assert(0);
#endif
	} else {
		LOG_DEBUG << "Using CPU routines for Vector-Vector convolution";
		transposer .reset(new  Transposer_CPU(dim_x, dim_y, dim_z, exp_x, exp_y, exp_z));
		transformer.reset(new Transformer_CPU(dim_x, dim_y, dim_z, exp_x, exp_y, exp_z));
		setupInverseTransforms();
	}

	// Setup lhs tensor field
	TensorFieldSetup setup(3, dim_x, dim_y, dim_z, exp_x, exp_y, exp_z);
	ComplexMatrix lhs2 = setup.transformTensorField(lhs);
	Matrix *foo1[3] = {&S.re[0], &S.re[1], &S.re[2]};
	Matrix *foo2[3] = {&S.im[0], &S.im[1], &S.im[2]};
	setup.unpackTransformedTensorField_xyz_to_zxy(lhs2, foo1, foo2);
}

VectorVectorConvolution_FFT::~VectorVectorConvolution_FFT()
{
	if (use_cuda) {
#ifdef HAVE_CUDA
		cufftDestroy(plan_x_c2r_cuda);
		cufftDestroy(plan_y_inv_cuda);
#else
		assert(0);
#endif
	} else {
		fftw_destroy_plan(plan_x_c2r);
		fftw_destroy_plan(plan_y_inv);
	}
}

void VectorVectorConvolution_FFT::execute(const VectorMatrix &rhs, VectorMatrix &res)
{
	assert(rhs.dimX() == dim_x && rhs.dimY() == dim_y && rhs.dimZ() == dim_z);
	assert(res.dimX() == dim_x && res.dimY() == dim_y && res.dimZ() == dim_z);

	if (!use_cuda) {
		Matrix::rw_accessor s1x_acc(s1.M[0]), s1y_acc(s1.M[1]), s1z_acc(s1.M[2]);
		Matrix::rw_accessor s2x_acc(s2.M[0]), s2y_acc(s2.M[1]), s2z_acc(s2.M[2]);
		double *s1x = s1x_acc.ptr(), *s1y = s1y_acc.ptr(), *s1z = s1z_acc.ptr();
		double *s2x = s2x_acc.ptr(), *s2y = s2y_acc.ptr(), *s2z = s2z_acc.ptr();

		transposer->copy_pad(rhs, s1x, s1y, s1z);
		transformer->transform_forward_x(s1x);
		transformer->transform_forward_x(s1y);
		transformer->transform_forward_x(s1z);
		transposer->transpose_zeropad_yzx(s1x, s1y, s1z, s2x, s2y, s2z);
		transformer->transform_forward_y(s2x);
		transformer->transform_forward_y(s2y);
		transformer->transform_forward_y(s2z);
		transposer->transpose_zeropad_zxy(s2x, s2y, s2z, s1x, s1y, s1z);
		transformer->transform_forward_z(s1x);
		transformer->transform_forward_z(s1y);
		transformer->transform_forward_z(s1z);
		calculate_multiplication(s1x /*inout*/, s1y /*in*/, s1z /*in*/);

		transformer->transform_inverse_z(s1x);

		// Back transform (except inverse Z-FFT) is different... ("+R")

		//transposer->transpose_unpad_yzx(s1x, s2x);
		cpu_transpose_unpad_c2c(
			exp_z, exp_x/2+1, exp_y,
			dim_z+R, 
			s1x, s2x
		);

		//transformer->transform_inverse_y(s2x);
		fftw_execute_dft(plan_y_inv, (fftw_complex*)s2x, (fftw_complex*)s2x);

		//transposer->transpose_unpad_xyz(s2x, s1x);
		cpu_transpose_unpad_c2c(
			exp_y, dim_z+R, exp_x/2+1, 
			dim_y+R, 
			s2x, s1x
		);

		//transformer->transform_inverse_x(s1x);
		fftw_execute_dft_c2r(plan_x_c2r, (fftw_complex*)s1x, (double*)s1x);

		//transposer->copy_unpad(s1x, res);
		cpu_copy_unpad_r2r(
			exp_x, dim_y+R, dim_z+R, 
			dim_x+R, 
			s1x, s2x
		);

		// s2x has dimensions (dim_x+R, dim_y+R, dim_z+R) now.
		gradient_cpu(delta_x, delta_y, delta_z, s2x /*in*/, res /*gradient out*/);
	} else { // cuda
#ifdef HAVE_CUDA
		// slightly different on GPU: x-transforms are out-of-place.
		Matrix::cu32_accessor s1x_acc(s1.M[0]), s1y_acc(s1.M[1]), s1z_acc(s1.M[2]);
		Matrix::cu32_accessor s2x_acc(s2.M[0]), s2y_acc(s2.M[1]), s2z_acc(s2.M[2]);
		float *s1x = s1x_acc.ptr(), *s1y = s1y_acc.ptr(), *s1z = s1z_acc.ptr();
		float *s2x = s2x_acc.ptr(), *s2y = s2y_acc.ptr(), *s2z = s2z_acc.ptr();

		Matrix::const_cu32_accessor Sx_re_acc(S.re[0]); const float *Sx_re = Sx_re_acc.ptr();
		Matrix::const_cu32_accessor Sy_re_acc(S.re[1]); const float *Sy_re = Sy_re_acc.ptr();
		Matrix::const_cu32_accessor Sz_re_acc(S.re[2]); const float *Sz_re = Sz_re_acc.ptr();
		Matrix::const_cu32_accessor Sx_im_acc(S.im[0]); const float *Sx_im = Sx_im_acc.ptr();
		Matrix::const_cu32_accessor Sy_im_acc(S.im[1]); const float *Sy_im = Sy_im_acc.ptr();
		Matrix::const_cu32_accessor Sz_im_acc(S.im[2]); const float *Sz_im = Sz_im_acc.ptr();

		transposer_cuda->copy_pad(rhs, s1x, s1y, s1z);
		transformer_cuda->transform_forward_x(s1x, s2x);
		transformer_cuda->transform_forward_x(s1y, s2y);
		transformer_cuda->transform_forward_x(s1z, s2z);
		transposer_cuda->transpose_zeropad_yzx(s2x, s2y, s2z, s1x, s1y, s1z);
		transformer_cuda->transform_forward_y(s1x);
		transformer_cuda->transform_forward_y(s1y);
		transformer_cuda->transform_forward_y(s1z);
		transposer_cuda->transpose_zeropad_zxy(s1x, s1y, s1z, s2x, s2y, s2z);
		transformer_cuda->transform_forward_z(s2x);
		transformer_cuda->transform_forward_z(s2y);
		transformer_cuda->transform_forward_z(s2z);

		//calculate_multiplication_cuda(s2x, s2y, s2z);
		cuda_multiplication_scalar_product(
			(exp_x/2+1) * exp_y * exp_z,
			Sx_re, Sy_re, Sz_re,
			Sx_im, Sy_im, Sz_im,
			s2x, s2y, s2z			
		);

		transformer_cuda->transform_inverse_z(s2x);

		// Back transform (except inverse Z-FFT) is different... ("+R")
		
		//transposer_cuda->transpose_unpad_yzx(s2x, s2y, s2z, s1x, s1y, s1z);
		cuda_transpose_unpad_c2c(
			exp_z, exp_x/2+1, exp_y,
			dim_z+R, 
			s2x, s1x
		);

		//transformer_cuda->transform_inverse_y(s1x);
		checkCufftSuccess(cufftExecC2C(plan_y_inv_cuda, (cufftComplex*)s1x, (cufftComplex*)s1x, CUFFT_INVERSE));

		//transposer_cuda->transpose_unpad_xyz(s1x, s1y, s1z, s2x, s2y, s2z);
		cuda_transpose_unpad_c2c(
			exp_y, dim_z+R, exp_x/2+1, 
			dim_y+R, 
			s1x, s2x
		);

		//transformer_cuda->transform_inverse_x(s2x, s1x);
		checkCufftSuccess(cufftExecC2R(plan_x_c2r_cuda, (cufftComplex*)s2x, (cufftReal*)s1x));

		//transposer_cuda->copy_unpad(s1x, s1y, s1z, res);
		cuda_copy_unpad_r2r(
			exp_x, dim_y+R, dim_z+R, 
			dim_x+R, 
			s1x, s2x
		);

		gradient_cuda(delta_x, delta_y, delta_z, s2x /*in*/, res /*gradient out*/);
#else
		assert(0);
#endif
	}
}

void VectorVectorConvolution_FFT::setupInverseTransforms()
{
	if (use_cuda) {
#ifdef HAVE_CUDA
		// Create CUFFT plans
		cufftResult res;
		
		res = cufftPlan1d(&plan_x_c2r_cuda, exp_x /*transform size*/, CUFFT_C2R, (dim_y+R)*(dim_z+R) /*batch size*/);
		if (res != CUFFT_SUCCESS) throw std::runtime_error("Error initializing cufft plan (4)");

		res = cufftPlan1d(&plan_y_inv_cuda, exp_y /*transform size*/, CUFFT_C2C, (exp_x/2+1)*(dim_z+R) /*batch size*/);
		if (res != CUFFT_SUCCESS) throw std::runtime_error("Error initializing cufft plan (5)");
#else
		assert(0);
#endif
	} else {
		Matrix::rw_accessor s1x_acc(s1.M[0]);
		double *s1x = s1x_acc.ptr();

		fftw_iodim dims, loop;

		// X-Transform: ((dim_y+R)*(dim_z+R)) x 1d-C2C-FFT (length: exp_x) in x-direction, in-place transform
		dims.n = exp_x;
		dims.is = 1;
		dims.os = 1;
		
		loop.n = (dim_y+R)*(dim_z+R);
		loop.is = exp_x/2+1;
		loop.os = exp_x;

		plan_x_c2r = fftw_plan_guru_dft_c2r(
			1, &dims, 
			1, &loop, 
			(fftw_complex*)s1x, 
			(      double*)s1x,
			fftw_strategy
		);
		assert(plan_x_c2r);

		// Y-Transform: ((dim_z+R)*exp_x/2+1) x 1d-C2C-FFT (length: exp_y) in x-direction, in-place transform
		dims.n = exp_y;
		dims.is = 1;
		dims.os = 1;
		
		loop.n = (dim_z+R)*(exp_x/2+1);
		loop.is = exp_y;
		loop.os = exp_y;

		plan_y_inv = fftw_plan_guru_dft(
			1, &dims,
			1, &loop,
			(fftw_complex*)s1x, // in
			(fftw_complex*)s1x, // out (-> in-place transform)
			FFTW_BACKWARD,
			fftw_strategy
		);
		assert(plan_y_inv);
	}
}

// Complex multiplication: res = a*b + c*d + e*f
template <typename T>
inline static void mul3(
	T &res_r, T &res_i,
	T ar, T ai,
	T br, T bi,
	T cr, T ci,
	T dr, T di,
	T er, T ei,
	T fr, T fi)
{
	// a*b
	res_r = ar*br - ai*bi; res_i = ai*br + ar*bi;
	// c*d
	res_r += cr*dr - ci*di; res_i += ci*dr + cr*di;
	// e*f
	res_r += er*fr - ei*fi; res_i += ei*fr + er*fi;
}

void VectorVectorConvolution_FFT::calculate_multiplication(double *inout_x, double *in_y, double *in_z)
{
	Matrix::ro_accessor Sx_re_acc(S.re[0]), Sx_im_acc(S.im[0]);
	Matrix::ro_accessor Sy_re_acc(S.re[1]), Sy_im_acc(S.im[1]);
	Matrix::ro_accessor Sz_re_acc(S.re[2]), Sz_im_acc(S.im[2]);
	const double *Sx_re = Sx_re_acc.ptr(), *Sx_im = Sx_im_acc.ptr();
	const double *Sy_re = Sy_re_acc.ptr(), *Sy_im = Sy_im_acc.ptr();
	const double *Sz_re = Sz_re_acc.ptr(), *Sz_im = Sz_im_acc.ptr();

	const int num_elements = (exp_x/2+1) * exp_y * exp_z;
	for (int n=0; n<num_elements; ++n) {
		const int m = n*2;

		const double Mx_re = inout_x[m+0], Mx_im = inout_x[m+1];
		const double My_re =    in_y[m+0], My_im =    in_y[m+1];
		const double Mz_re =    in_z[m+0], Mz_im =    in_z[m+1];

		double phi_re, phi_im;
		mul3<double>(phi_re, phi_im,             // phi = 
		     Mx_re, Mx_im, Sx_re[n], Sx_im[n],   //       Sx*Mx
		     My_re, My_im, Sy_re[n], Sy_im[n],   //     + Sy*My
		     Mz_re, Mz_im, Sz_re[n], Sz_im[n]);  //     + Sz*Mz

		// output
		inout_x[m+0] = phi_re;
		inout_x[m+1] = phi_im;
	}
}
