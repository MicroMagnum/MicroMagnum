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

#include "transpose_3d.h"

#include <cassert>
#include <iostream>
using namespace std;

#include "kernels.h"

void ng_cuda_transpose_zeropad_c2c_3d( // xyz -> Yzx
	cudaStream_t s0, cudaStream_t s1,
	int dim_x, int dim_y, int dim_z, // input size
	int exp_y, // exp_y >= dim_y
	const float * in, // size: dim_x * dim_y * dim_z
	      float *out  // size: exp_y * dim_z * dim_x (zero-padded in x-direction from dim_y -> exp_y)
)
{
	configure_transpose_kernels();

	// Input size: dim_x * dim_y * dim_z
	const int in_stride_x = 1;
	const int in_stride_y = in_stride_x * dim_x;
	const int in_stride_z = in_stride_y * dim_y;

	// Output size: exp_y * dim_z * dim_x
	const int out_stride_x = 1;
	const int out_stride_y = out_stride_x * exp_y;
	const int out_stride_z = out_stride_y * dim_z;


	// I. Transpose part
	{
		// Setup grid and block sizes
		const dim3 threads = dim3(8, 8, 8);
		      dim3 grid = dim3((dim_x-1)/threads.x+1, (dim_y-1)/threads.y+1, (dim_z-1)/threads.z+1);

		const int gridDim_y = grid.y;
		const int gridDim_z = grid.z;
		if (grid.z > 1) {
			grid.y *= grid.z;
			grid.z = 1;
		}

		kernel_rotate_left_3d<<<grid, threads, 0, s0>>>(dim_x, dim_y, dim_z, in, in_stride_y, in_stride_z, out, out_stride_y, out_stride_z, gridDim_y, gridDim_z);
		checkCudaLastError("kernel_rotate_left_3d execution failed");
	}

	// II. Zeropad part. 
	if (exp_y > dim_y) {
		const dim3 threads = dim3(16, 8, 8);
		      dim3 grid    = dim3((exp_y-dim_y-1)/threads.x+1, (dim_z-1)/threads.y+1, (dim_x-1)/threads.z+1);

		const int gridDim_y = grid.y;
		const int gridDim_z = grid.z;
		if (grid.z > 1) {
			grid.y *= grid.z;
			grid.z = 1;
		}

		kernel_clear_3d<<<grid, threads, 0, s1>>>(exp_y - dim_y, dim_z, dim_x, out + 2*dim_y, out_stride_y, out_stride_z, gridDim_y, gridDim_z);
		checkCudaLastError("kernel_clear_3d execution failed");
	}
	CUDA_THREAD_SYNCHRONIZE();
}

void ng_cuda_transpose_unpad_c2c_3d( // XYZ -> ZxY
	cudaStream_t s0,
	int dim_x, int dim_y, int dim_z, // input size
	int red_x, // red_x <= dim_x
	const float *in,   // size: dim_x * dim_y * dim_z
	      float *out   // size: dim_z * red_x * dim_y (cut in x-direction from dim_x->red_x)
)
{
	configure_transpose_kernels();

	const int  in_stride_x = 1;
	const int  in_stride_y =  in_stride_x * dim_x;
	const int  in_stride_z =  in_stride_y * dim_y;

	const int out_stride_x = 1;
	const int out_stride_y = out_stride_x * dim_z;
	const int out_stride_z = out_stride_y * red_x;

	const dim3 threads = dim3(8, 8, 8);
	      dim3 grid    = dim3((red_x-1)/threads.x+1, (dim_y-1)/threads.y+1, (dim_z-1)/threads.z+1);

	const int gridSize_y = grid.y;
	const int gridSize_z = grid.z;
	if (grid.z > 1) {
		grid.y *= grid.z;
		grid.z = 1;
	}

	kernel_rotate_right_3d<<<grid, threads, 0, s0>>>(dim_x, dim_y, dim_z, in, in_stride_y, in_stride_z, out, out_stride_y, out_stride_z, gridSize_y, gridSize_z);
	checkCudaLastError("kernel_transpose_2d execution failed");

	CUDA_THREAD_SYNCHRONIZE();
}
