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

#include "Transposer_CUDA.h"

#include "matrix/device/cuda_tools.h"

#include "cuda_copy_pad.h"
#include "cuda_copy_unpad.h"
#include "cuda_transpose_zeropad.h"
#include "cuda_transpose_unpad.h"

#include "Magneto.h"

Transposer_CUDA::Transposer_CUDA(int dim_x, int dim_y, int dim_z, int exp_x, int exp_y, int exp_z)
	: dim_x(dim_x), dim_y(dim_y), dim_z(dim_z), exp_x(exp_x), exp_y(exp_y), exp_z(exp_z)
{
}

Transposer_CUDA::~Transposer_CUDA()
{
}

void Transposer_CUDA::copy_pad(const VectorMatrix &M, float *out_x, float *out_y, float *out_z)
{
	// Ifdef HAVE_CUDA_64, we directly support input matrices that
	// are stored with 64 bit precision on the GPU.
#ifdef HAVE_CUDA_64
	const bool M_is_cuda64_bit = M.isCached(2); // 0 = CPU device, 2 = CUDA_64 device
	if (M_is_cuda64_bit) {
		VectorMatrix::const_cu64_accessor M_acc(M);
		// xyz, M -> s1
		cuda_copy_pad_r2r(dim_x, dim_y, dim_z, exp_x, M_acc.ptr_x(), M_acc.ptr_y(), M_acc.ptr_z(), out_x, out_y, out_z);
	} 
	else
#endif
	{
		VectorMatrix::const_cu32_accessor M_acc(M);
		// xyz, M -> s1
		cuda_copy_pad_r2r(dim_x, dim_y, dim_z, exp_x, M_acc.ptr_x(), M_acc.ptr_y(), M_acc.ptr_z(), out_x, out_y, out_z);
	}
}

void Transposer_CUDA::copy_unpad(const float *in_x, const float *in_y, const float *in_z, VectorMatrix &H)
{	
	// Ifdef HAVE_CUDA_64 and isCuda64Enabled(), we directly store output matrices on the GPU with 64 bit precision.
#ifdef HAVE_CUDA_64
	if (isCuda64Enabled()) {
		// xyz, s1 -> H
		VectorMatrix::cu64_accessor H_acc(H);
		cuda_copy_unpad_r2r(exp_x, dim_y, dim_z, dim_x, in_x, in_y, in_z, H_acc.ptr_x(), H_acc.ptr_y(), H_acc.ptr_z());
	}
	else
#endif
	{
		// xyz, s1 -> H
		VectorMatrix::cu32_accessor H_acc(H);
		cuda_copy_unpad_r2r(exp_x, dim_y, dim_z, dim_x, in_x, in_y, in_z, H_acc.ptr_x(), H_acc.ptr_y(), H_acc.ptr_z());
	}
}

void Transposer_CUDA::transpose_zeropad_yzx(const float *in_x, const float *in_y, const float *in_z, float *out_x, float *out_y, float *out_z)
{
	// xyz->yzx
	cuda_transpose_zeropad_c2c(
		exp_x/2+1, dim_y, dim_z, 
		exp_y,
		in_x, in_y, in_z,
		out_x, out_y, out_z
	);
}

void Transposer_CUDA::transpose_zeropad_zxy(const float *in_x, const float *in_y, const float *in_z, float *out_x, float *out_y, float *out_z)
{
	// yzx->zxy
	cuda_transpose_zeropad_c2c(
		exp_y, dim_z, exp_x/2+1,
		exp_z,
		in_x, in_y, in_z,
		out_x, out_y, out_z
	);
}

void Transposer_CUDA::transpose_unpad_yzx(const float *in_x, const float *in_y, const float *in_z, float *out_x, float *out_y, float *out_z)
{
	// zxy->yzx
	cuda_transpose_unpad_c2c(
		exp_z, exp_x/2+1, exp_y,
		dim_z,
		in_x, in_y, in_z,
		out_x, out_y, out_z
	);
}

void Transposer_CUDA::transpose_unpad_xyz(const float *in_x, const float *in_y, const float *in_z, float *out_x, float *out_y, float *out_z)
{
	// yzx->xyz
	cuda_transpose_unpad_c2c(
		exp_y, dim_z, exp_x/2+1,
		dim_y,
		in_x, in_y, in_z,
		out_x, out_y, out_z
	);
}
