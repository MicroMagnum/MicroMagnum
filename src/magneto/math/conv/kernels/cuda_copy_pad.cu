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

#include "cuda_copy_pad.h"

#include <cuda.h>
#include "matrix/device/cuda_tools.h"

#include <cassert>

template <typename real>
__global__
void kernel_copy_pad_r2r_2d(
	int dim_x, int dim_y, 
	int exp_x,
	const  real * in_x, const  real * in_y, const  real * in_z, 
	      float *out_x,       float *out_y,       float *out_z)
{
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (y < dim_y) {
		const int out_idx = 1*(x + y*exp_x);
		if (x < dim_x) {
			const int in_idx = 1*(x + y*dim_x);
			out_x[out_idx] = in_x[in_idx];
			out_y[out_idx] = in_y[in_idx];
			out_z[out_idx] = in_z[in_idx];
		} else if (x < exp_x) {
			out_x[out_idx] = 0.0;
			out_y[out_idx] = 0.0;
			out_z[out_idx] = 0.0;
		}
	}
}

template <typename real>
__global__
void kernel_copy_pad_r2r_2d_scalar(
	int dim_x, int dim_y, 
	int exp_x,
	const  real * in_x,
	      float *out_x)
{
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (y < dim_y) {
		const int out_idx = 1*(x + y*exp_x);
		if (x < dim_x) {
			const int in_idx = 1*(x + y*dim_x);
			out_x[out_idx] = in_x[in_idx];
		} else if (x < exp_x) {
			out_x[out_idx] = 0.0;
		}
	}
}

template <typename real>
__global__
void kernel_copy_pad_r2r_3d(
	int dim_x, int dim_y, int dim_z,
	int exp_x,
	const real *in, float *out)
{
	// This thread block copies the row 
	// from (0, row_y, row_z) to (dim_x, row_y, row_z) and pads up to (exp_x, row_y, row_z)
	const int row_y = blockIdx.x;
	const int row_z = blockIdx.y;

	const int tidx = threadIdx.x; // thread idx in block
	const int tnum = blockDim.x;  // numbers of threads in block

	in  += row_y*dim_x + row_z*dim_x*dim_y;
	out += row_y*exp_x + row_z*exp_x*dim_y;

	// copy..
	for (int i=tidx; i<dim_x; i+=tnum) {
		out[i] = in[i];
	}

	// pad..
	for (int i=dim_x+tidx; i<exp_x; i+=tnum) {
		out[i] = 0.0;
	}
}

////////////////////////////////////////////////////////////////////////////////

template <typename real>
void cuda_copy_pad_r2r_impl(
	int dim_x, int dim_y, int dim_z,
	int exp_x,
	const  real * in_x, const  real * in_y, const  real * in_z, 
	      float *out_x,       float *out_y,       float *out_z)
{
	if (dim_z == 1) {
		const dim3 threads_per_block(16, 16, 1);
		const dim3 num_blocks((exp_x+15) / 16, (dim_y+15) / 16, 1);

		kernel_copy_pad_r2r_2d<real><<<num_blocks, threads_per_block>>>(
			dim_x, dim_y, 
			exp_x, 
			 in_x,  in_y,  in_z, 
			out_x, out_y, out_z
		);
		checkCudaLastError("kernel_copy_pad_r2r_2d() execution failed");
	} else {
		const dim3 threads_per_block(128, 1, 1);
		const dim3 num_blocks(dim_y, dim_z, 1);

		const  real  *in[3] = { in_x,  in_y,  in_z};
		      float *out[3] = {out_x, out_y, out_z};

		for (int c=0; c<3; ++c) {
			kernel_copy_pad_r2r_3d<real><<<num_blocks, threads_per_block>>>(
				dim_x, dim_y, dim_z,
				exp_x,
				in[c], out[c]
			);
			checkCudaLastError("kernel_copy_pad_r2r_3d() execution failed");
		}
	}

	CUDA_THREAD_SYNCHRONIZE();
}

template <typename real>
void cuda_copy_pad_r2r_impl(
	int dim_x, int dim_y, int dim_z,
	int exp_x,
	const  real * in_x,
	      float *out_x)
{
	if (dim_z == 1) {
		const dim3 threads_per_block(16, 16, 1);
		const dim3 num_blocks((dim_x+15) / 16, (dim_y+15) / 16, 1);

		kernel_copy_pad_r2r_2d_scalar<real><<<num_blocks, threads_per_block>>>(
			dim_x, dim_y, 
			exp_x, 
			 in_x, out_x
		);
		checkCudaLastError("kernel_copy_pad_r2r_2d_scalar() execution failed");
	} else {
		const dim3 threads_per_block(128, 1, 1);
		const dim3 num_blocks(dim_y, dim_z, 1);

		kernel_copy_pad_r2r_3d<real><<<num_blocks, threads_per_block>>>(
			dim_x, dim_y, dim_z,
			exp_x,
			in_x, out_x
		);
		checkCudaLastError("kernel_copy_pad_r2r_3d() execution failed");
	}

	CUDA_THREAD_SYNCHRONIZE();
}

//////////////////////////////////////////////////////////////////////////////// 

void cuda_copy_pad_r2r(
	int dim_x, int dim_y, int dim_z,
	int exp_x,
	const float * in_x,
	      float *out_x)
{
	cuda_copy_pad_r2r_impl<float>(dim_x, dim_y, dim_z, exp_x, in_x, out_x);
}

void cuda_copy_pad_r2r(
	int dim_x, int dim_y, int dim_z,
	int exp_x,
	const double * in_x,
	       float *out_x)
{
	cuda_copy_pad_r2r_impl<double>(dim_x, dim_y, dim_z, exp_x, in_x, out_x);
}

void cuda_copy_pad_r2r(
	int dim_x, int dim_y, int dim_z,
	int exp_x,
	const float * in_x, const float * in_y, const float * in_z, 
	      float *out_x,       float *out_y,       float *out_z)
{
	cuda_copy_pad_r2r_impl<float>(dim_x, dim_y, dim_z, exp_x, in_x, in_y, in_z, out_x, out_y, out_z);
}

void cuda_copy_pad_r2r(
	int dim_x, int dim_y, int dim_z,
	int exp_x,
	const double * in_x, const double * in_y, const double * in_z,
	       float *out_x,        float *out_y,        float *out_z)
{
	cuda_copy_pad_r2r_impl<double>(dim_x, dim_y, dim_z, exp_x, in_x, in_y, in_z, out_x, out_y, out_z);
}
