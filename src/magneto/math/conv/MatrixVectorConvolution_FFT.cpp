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

#include "MatrixVectorConvolution_FFT.h"

#include "TensorFieldSetup.h"

#include "Magneto.h"
#include "Benchmark.h"
#include "Logger.h"

#include "kernels/Transposer_CPU.h"
#include "kernels/Transformer_CPU.h"
#include "kernels/cpu_multiplication.h"
#ifdef HAVE_CUDA
#include <cuda_runtime.h> // cudaThreadSynchronize
#include "kernels/Transposer_CUDA.h"
#include "kernels/Transformer_CUDA.h"
#include "kernels/cuda_multiplication.h"
#endif

MatrixVectorConvolution_FFT::MatrixVectorConvolution_FFT(int dim_x, int dim_y, int dim_z, int exp_x, int exp_y, int exp_z)
	: dim_x(dim_x), dim_y(dim_y), dim_z(dim_z), exp_x(exp_x), exp_y(exp_y), exp_z(exp_z)
{
	// Enable CUDA?
#ifdef HAVE_CUDA
	use_cuda = isCudaEnabled();
#else
	use_cuda = false;
#endif

	// 2-D problem?
	is_2d = (dim_z == 1) && (exp_z == 1);

	// Allocate buffers
	for (int e=0; e<3; ++e) {
		s1.M[e] = Matrix(Shape(2, exp_x/2+1, exp_y, exp_z));
		s2.M[e] = Matrix(Shape(2, exp_x/2+1, exp_y, exp_z));
	}

	// Initializer transposer & transformer
	if (use_cuda) {
#ifdef HAVE_CUDA
		LOG_DEBUG << "Using CUDA routines for Matrix-Vector convolution";
		transposer_cuda .reset(new  Transposer_CUDA(dim_x, dim_y, dim_z, exp_x, exp_y, exp_z));
		transformer_cuda.reset(new Transformer_CUDA(dim_x, dim_y, dim_z, exp_x, exp_y, exp_z));
#else
		assert(0);
#endif
	} else {
		LOG_DEBUG << "Using CPU routines for Matrix-Vector convolution";
		transposer .reset(new  Transposer_CPU(dim_x, dim_y, dim_z, exp_x, exp_y, exp_z));
		transformer.reset(new Transformer_CPU(dim_x, dim_y, dim_z, exp_x, exp_y, exp_z));
	}

	// Setup profiling
	if (is_2d) {
		Benchmark::inst().setDescription("conv2d", "convolution total (2d)");
		Benchmark::inst().setDescription("conv2d.pad", "padding");
		Benchmark::inst().setDescription("conv2d.fft", "forward FFT total");
		Benchmark::inst().setDescription("conv2d.fft.x", "partial FFT x");
		Benchmark::inst().setDescription("conv2d.fft.transpose", "transpose for fft");
		Benchmark::inst().setDescription("conv2d.fft.y", "partial FFT y");
		Benchmark::inst().setDescription("conv2d.mult", "multipliation");
		Benchmark::inst().setDescription("conv2d.ifft", "inverse FFT total");
		Benchmark::inst().setDescription("conv2d.ifft.y", "partial FFT y");
		Benchmark::inst().setDescription("conv2d.ifft.transpose", "transpose for ifft");
		Benchmark::inst().setDescription("conv2d.ifft.x", "partial FFT x");
		Benchmark::inst().setDescription("conv2d.unpad", "unpadding");
	} else {
		Benchmark::inst().setDescription("conv3d", "convolution total (3d)");
		Benchmark::inst().setDescription("conv3d.pad", "padding");
		Benchmark::inst().setDescription("conv3d.fft", "forward FFT total");
		Benchmark::inst().setDescription("conv3d.fft.x", "partial FFT x");
		Benchmark::inst().setDescription("conv3d.fft.transpose1", "transpose1 for fft");
		Benchmark::inst().setDescription("conv3d.fft.y", "partial FFT y");
		Benchmark::inst().setDescription("conv3d.fft.z", "partial FFT z");
		Benchmark::inst().setDescription("conv3d.fft.transpose2", "transpose1 for fft");
		Benchmark::inst().setDescription("conv3d.mult", "multipliation");
		Benchmark::inst().setDescription("conv3d.ifft", "inverse FFT total");
		Benchmark::inst().setDescription("conv3d.ifft.z", "partial FFT z");
		Benchmark::inst().setDescription("conv3d.ifft.transpose2", "transpose2 for ifft");
		Benchmark::inst().setDescription("conv3d.ifft.y", "partial FFT y");
		Benchmark::inst().setDescription("conv3d.ifft.transpose1", "transpose1 for ifft");
		Benchmark::inst().setDescription("conv3d.ifft.x", "partial FFT x");
		Benchmark::inst().setDescription("conv3d.unpad", "unpadding");
	}
}

MatrixVectorConvolution_FFT::~MatrixVectorConvolution_FFT()
{
}

void MatrixVectorConvolution_FFT::execute(const VectorMatrix &rhs, VectorMatrix &res)
{
	assert(rhs.dimX() == dim_x && rhs.dimY() == dim_y && rhs.dimZ() == dim_z);
	assert(res.dimX() == dim_x && res.dimY() == dim_y && res.dimZ() == dim_z);

	if (!use_cuda) {
		Matrix::rw_accessor s1x_acc(s1.M[0]), s1y_acc(s1.M[1]), s1z_acc(s1.M[2]);
		Matrix::rw_accessor s2x_acc(s2.M[0]), s2y_acc(s2.M[1]), s2z_acc(s2.M[2]);
		double *s1x = s1x_acc.ptr(), *s1y = s1y_acc.ptr(), *s1z = s1z_acc.ptr();
		double *s2x = s2x_acc.ptr(), *s2y = s2y_acc.ptr(), *s2z = s2z_acc.ptr();

		if (is_2d) {
			TIC("conv2d");
				TIC("conv2d.pad");
					transposer->copy_pad(rhs, s1x, s1y, s1z);
				TOC("conv2d.pad");

				TIC("conv2d.fft");
					TIC("conv2d.fft.x");
						transformer->transform_forward_x(s1x);
						transformer->transform_forward_x(s1y);
						transformer->transform_forward_x(s1z);
					TOC("conv2d.fft.x");

					TIC("conv2d.fft.transpose");
						transposer->transpose_zeropad_yzx(s1x, s1y, s1z, s2x, s2y, s2z);
					TOC("conv2d.fft.transpose");

					TIC("conv2d.fft.y");
						transformer->transform_forward_y(s2x);
						transformer->transform_forward_y(s2y);
						transformer->transform_forward_y(s2z);
					TOC("conv2d.fft.y");
				TOC("conv2d.fft");

				TIC("conv2d.mult");
					calculate_multiplication(s2x, s2y, s2z);
				TOC("conv2d.mult");

				TIC("conv2d.ifft");
					TIC("conv2d.ifft.y");
						transformer->transform_inverse_y(s2x);
						transformer->transform_inverse_y(s2y);
						transformer->transform_inverse_y(s2z);
					TOC("conv2d.ifft.y");

					TIC("conv2d.ifft.transpose");
						transposer->transpose_unpad_xyz(s2x, s2y, s2z, s1x, s1y, s1z);
					TOC("conv2d.ifft.transpose");

					TIC("conv2d.ifft.x");
						transformer->transform_inverse_x(s1x);
						transformer->transform_inverse_x(s1y);
						transformer->transform_inverse_x(s1z);
					TOC("conv2d.ifft.x");
				TOC("conv2d.ifft");

				TIC("conv2d.unpad");
				transposer->copy_unpad(s1x, s1y, s1z, res);
				TOC("conv2d.unpad");
			TOC("conv2d");
		} else {
			TIC("conv3d");
				TIC("conv3d.pad");
					transposer->copy_pad(rhs, s1x, s1y, s1z);
				TOC("conv3d.pad");

				TIC("conv3d.fft");
					TIC("conv3d.fft.x");
						transformer->transform_forward_x(s1x);
						transformer->transform_forward_x(s1y);
						transformer->transform_forward_x(s1z);
					TOC("conv3d.fft.x");

					TIC("conv3d.fft.transpose1");
						transposer->transpose_zeropad_yzx(s1x, s1y, s1z, s2x, s2y, s2z);
					TOC("conv3d.fft.transpose1");

					TIC("conv3d.fft.y");
						transformer->transform_forward_y(s2x);
						transformer->transform_forward_y(s2y);
						transformer->transform_forward_y(s2z);
					TOC("conv3d.fft.y");

					TIC("conv3d.fft.transpose2");
						transposer->transpose_zeropad_zxy(s2x, s2y, s2z, s1x, s1y, s1z);
					TOC("conv3d.fft.transpose2");

					TIC("conv3d.fft.z");
						transformer->transform_forward_z(s1x);
						transformer->transform_forward_z(s1y);
						transformer->transform_forward_z(s1z);
					TOC("conv3d.fft.z");
				TOC("conv3d.fft");

				TIC("conv3d.mult");
					calculate_multiplication(s1x, s1y, s1z);
				TOC("conv3d.mult");

				TIC("conv3d.ifft");
					TIC("conv3d.ifft.z");
						transformer->transform_inverse_z(s1x);
						transformer->transform_inverse_z(s1y);
						transformer->transform_inverse_z(s1z);
					TOC("conv3d.ifft.z");

					TIC("conv3d.ifft.transpose2");
						transposer->transpose_unpad_yzx(s1x, s1y, s1z, s2x, s2y, s2z);
					TOC("conv3d.ifft.transpose2");

					TIC("conv3d.ifft.y");
						transformer->transform_inverse_y(s2x);
						transformer->transform_inverse_y(s2y);
						transformer->transform_inverse_y(s2z);
					TOC("conv3d.ifft.y");

					TIC("conv3d.ifft.transpose1");
						transposer->transpose_unpad_xyz(s2x, s2y, s2z, s1x, s1y, s1z);
					TOC("conv3d.ifft.transpose1");

					TIC("conv3d.ifft.x");
						transformer->transform_inverse_x(s1x);
						transformer->transform_inverse_x(s1y);
						transformer->transform_inverse_x(s1z);
					TOC("conv3d.ifft.x");
				TOC("conv3d.ifft");

				TIC("conv3d.unpad");
					transposer->copy_unpad(s1x, s1y, s1z, res);
				TOC("conv3d.unpad");
			TOC("conv3d");
		}
	} else { // cuda
#ifdef HAVE_CUDA
		// slightly different on GPU: x-transforms are out-of-place.
		Matrix::cu32_accessor s1x_acc(s1.M[0]), s1y_acc(s1.M[1]), s1z_acc(s1.M[2]);
		Matrix::cu32_accessor s2x_acc(s2.M[0]), s2y_acc(s2.M[1]), s2z_acc(s2.M[2]);
		float *s1x = s1x_acc.ptr(), *s1y = s1y_acc.ptr(), *s1z = s1z_acc.ptr();
		float *s2x = s2x_acc.ptr(), *s2y = s2y_acc.ptr(), *s2z = s2z_acc.ptr();

		if (is_2d) {
			CUTIC("conv2d");
				CUTIC("conv2d.pad");
					transposer_cuda->copy_pad(rhs, s1x, s1y, s1z);
				CUTOC("conv2d.pad");

				CUTIC("conv2d.fft");
					CUTIC("conv2d.fft.x");
						transformer_cuda->transform_forward_x(s1x, s2x);
						transformer_cuda->transform_forward_x(s1y, s2y);
						transformer_cuda->transform_forward_x(s1z, s2z);
					CUTOC("conv2d.fft.x");

					CUTIC("conv2d.fft.transpose");
					transposer_cuda->transpose_zeropad_yzx(s2x, s2y, s2z, s1x, s1y, s1z);
					CUTOC("conv2d.fft.transpose");

					CUTIC("conv2d.fft.y");
						transformer_cuda->transform_forward_y(s1x);
						transformer_cuda->transform_forward_y(s1y);
						transformer_cuda->transform_forward_y(s1z);
					CUTOC("conv2d.fft.y");
				CUTOC("conv2d.fft");

				CUTIC("conv2d.mult");
					calculate_multiplication_cuda(s1x, s1y, s1z);
				CUTOC("conv2d.mult");

				CUTIC("conv2d.ifft");
					CUTIC("conv2d.ifft.y");
						transformer_cuda->transform_inverse_y(s1x);
						transformer_cuda->transform_inverse_y(s1y);
						transformer_cuda->transform_inverse_y(s1z);
					CUTOC("conv2d.ifft.y");

					CUTIC("conv2d.ifft.transpose");
						transposer_cuda->transpose_unpad_xyz(s1x, s1y, s1z, s2x, s2y, s2z);
					CUTOC("conv2d.ifft.transpose");

					CUTIC("conv2d.ifft.x");
						transformer_cuda->transform_inverse_x(s2x, s1x);
						transformer_cuda->transform_inverse_x(s2y, s1y);
						transformer_cuda->transform_inverse_x(s2z, s1z);
					CUTOC("conv2d.ifft.x");
				CUTOC("conv2d.ifft");

				CUTIC("conv2d.unpad");
					transposer_cuda->copy_unpad(s1x, s1y, s1z, res);
				CUTOC("conv2d.unpad");
			CUTOC("conv2d");
		} else {
			CUTIC("conv3d");
				CUTIC("conv3d.pad");
					transposer_cuda->copy_pad(rhs, s1x, s1y, s1z);
				CUTOC("conv3d.pad");

				CUTIC("conv3d.fft");
					CUTIC("conv3d.fft.x");
						transformer_cuda->transform_forward_x(s1x, s2x);
						transformer_cuda->transform_forward_x(s1y, s2y);
						transformer_cuda->transform_forward_x(s1z, s2z);
					CUTOC("conv3d.fft.x");

					CUTIC("conv3d.fft.transpose1");
						transposer_cuda->transpose_zeropad_yzx(s2x, s2y, s2z, s1x, s1y, s1z);
					CUTOC("conv3d.fft.transpose1");

					CUTIC("conv3d.fft.y");
						transformer_cuda->transform_forward_y(s1x);
						transformer_cuda->transform_forward_y(s1y);
						transformer_cuda->transform_forward_y(s1z);
					CUTOC("conv3d.fft.y");

					CUTIC("conv3d.fft.transpose2");
						transposer_cuda->transpose_zeropad_zxy(s1x, s1y, s1z, s2x, s2y, s2z);
					CUTOC("conv3d.fft.transpose2");

					CUTIC("conv3d.fft.z");
						transformer_cuda->transform_forward_z(s2x);
						transformer_cuda->transform_forward_z(s2y);
						transformer_cuda->transform_forward_z(s2z);
					CUTOC("conv3d.fft.z");
				CUTOC("conv3d.fft");

				CUTIC("conv3d.mult");
					calculate_multiplication_cuda(s2x, s2y, s2z);
				CUTOC("conv3d.mult");

				CUTIC("conv3d.ifft");
					CUTIC("conv3d.ifft.z");
						transformer_cuda->transform_inverse_z(s2x);
						transformer_cuda->transform_inverse_z(s2y);
						transformer_cuda->transform_inverse_z(s2z);
					CUTOC("conv3d.ifft.z");

					CUTIC("conv3d.ifft.transpose2");
						transposer_cuda->transpose_unpad_yzx(s2x, s2y, s2z, s1x, s1y, s1z);
					CUTOC("conv3d.ifft.transpose2");

					CUTIC("conv3d.ifft.y");
						transformer_cuda->transform_inverse_y(s1x);
						transformer_cuda->transform_inverse_y(s1y);
						transformer_cuda->transform_inverse_y(s1z);
					CUTOC("conv3d.ifft.y");

					CUTIC("conv3d.ifft.transpose1");
						transposer_cuda->transpose_unpad_xyz(s1x, s1y, s1z, s2x, s2y, s2z);
					CUTOC("conv3d.ifft.transpose1");

					CUTIC("conv3d.ifft.x");
						transformer_cuda->transform_inverse_x(s2x, s1x);
						transformer_cuda->transform_inverse_x(s2y, s1y);
						transformer_cuda->transform_inverse_x(s2z, s1z);
					CUTOC("conv3d.ifft.x");

					CUTIC("conv3d.unpad");
						transposer_cuda->copy_unpad(s1x, s1y, s1z, res);
					CUTOC("conv3d.unpad");
				CUTOC("conv3d.ifft");
			CUTOC("conv3d");
		}
#else
		assert(0);
#endif
	}
}
