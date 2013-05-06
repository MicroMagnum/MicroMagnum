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

#include "transpose_2d.h"

#include <cassert>

#include "kernels.h"

#include <iostream>
using namespace std;

void ng_cuda_transpose_zeropad_c2c_2d(
	cudaStream_t s0, cudaStream_t s1,
	int dim_x, int dim_y, // input size
	int exp_y,
	const float *in, // size: dim_x * dim_y
	      float *out // size: exp_y * dim_x (zero-padded in x-direction from dim_y -> exp_y)
)
{
	configure_transpose_kernels();

	const int  in_stride_x = 1;
	const int  in_stride_y = in_stride_x * dim_x;

	const int out_stride_x = 1;
	const int out_stride_y = out_stride_x * exp_y;

	// I. Transpose part
	{
		// Setup grid and block sizes
		const dim3 threads = dim3(16, 16, 1);
		const dim3 grid    = dim3((dim_x-1)/threads.x+1, (dim_y-1)/threads.y+1, 1);

		kernel_transpose_2d<<<grid, threads, 0, s0>>>(dim_x, dim_y, in, in_stride_y, out, out_stride_y);
		checkCudaLastError("kernel_transpose_2d execution failed");
	}

	// II. Zeropad part. 
	if (exp_y > dim_y) {
		const dim3 threads = dim3(16, 16, 1);
		const dim3 grid    = dim3(((exp_y-dim_y)-1)/threads.x+1, (dim_x-1)/threads.y+1, 1);

		kernel_clear_2d<<<grid, threads, 0, s1>>>(exp_y - dim_y, dim_x, out + 2*dim_y, out_stride_y);
		checkCudaLastError("kernel_clear_2d execution failed");
	}

	CUDA_THREAD_SYNCHRONIZE();
}

void ng_cuda_transpose_unpad_c2c_2d(
	cudaStream_t s0,
	int dim_x, int dim_y, // input size
	int red_x, // red_x <= dim_x
	const float * in, // size: dim_x * dim_y
	      float *out) // size: dim_y * red_x
{
	configure_transpose_kernels();

	const int  in_stride_x = 1;
	const int  in_stride_y = in_stride_x * dim_x;

	const int out_stride_x = 1;
	const int out_stride_y = out_stride_x * dim_y;

	const dim3 threads = dim3(16, 16, 1);
	const dim3 grid    = dim3((red_x-1)/threads.x+1, (dim_y-1)/threads.y+1, 1);

	kernel_transpose_2d<<<grid, threads, 0, s0>>>(red_x, dim_y, in, in_stride_y, out, out_stride_y);
	checkCudaLastError("kernel_transpose_2d execution failed");

	CUDA_THREAD_SYNCHRONIZE();
}

