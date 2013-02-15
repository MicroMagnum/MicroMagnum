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

#include "cuda_transpose_zeropad_3d.h"

#include <cuda.h>
#include "matrix/device/cuda_tools.h"

#include <cassert>
//#include <iostream>
//using namespace std;

static const bool use_naive_kernel = true;

__global__
void kernel_transpose_unpad_c2c_3d_naive(
	int dim_x, int dim_y, int dim_z,
	int red_x,
	const float *in, float *out,
	int logical_gridDim_y)
{
	const int blockIdx_x = blockIdx.x;
	const int blockIdx_y = blockIdx.y % logical_gridDim_y;
	const int blockIdx_z = blockIdx.y / logical_gridDim_y;

	const int x = blockIdx_x*8 + threadIdx.x;
	const int y = blockIdx_y*8 + threadIdx.y;
	const int z = blockIdx_z*8 + threadIdx.z;

	if (z < dim_y && y < red_x && x < dim_z) {
		const int in_stride_x = 2 * 1;
		const int in_stride_y = 2 * dim_x;
		const int in_stride_z = 2 * dim_x * dim_y;

		const int out_stride_x = 2 * 1;
		const int out_stride_y = 2 * dim_z;
		const int out_stride_z = 2 * dim_z * red_x;

		in  += z* in_stride_y + y* in_stride_x + x* in_stride_z;
		out += z*out_stride_z + y*out_stride_y + x*out_stride_x;

		out[0] = in[0];
		out[1] = in[1];
	}
}

void cuda_transpose_unpad_c2c_3d(
	int dim_x, int dim_y, int dim_z, // input size
	int red_x, // red_x <= dim_x
	const float *in_x, const float *in_y, const float *in_z, // size: dim_x * dim_y * dim_z
	      float *out_x, float *out_y, float *out_z)          // size: dim_z * red_x * dim_y (cut in x-direction [of input] from dim_x->red_x)
{
	if (use_naive_kernel) {
		dim3 threads_per_block(8, 8, 8);
		dim3 num_blocks((dim_z+7)/8+1, (red_x+7)/8+1, (dim_y+7/8+1));

		const int logical_gridDim_y = num_blocks.y;
		num_blocks.y *= num_blocks.z;
		num_blocks.z = 1;

		const float  *in[3] = { in_x,  in_y,  in_z};
		      float *out[3] = {out_x, out_y, out_z};

		for (int c=0; c<3; ++c) {
			kernel_transpose_unpad_c2c_3d_naive<<<num_blocks, threads_per_block>>>(
				dim_x, dim_y, dim_z,
				red_x,
				in[c], out[c],
				logical_gridDim_y
			);		
			checkCudaLastError("kernel_transpose_zeropad_c2c_2d execution failed");
		}
	} else {
		assert(0);
	}

	CUDA_THREAD_SYNCHRONIZE();
}

// XYZ -> ZxY, scalar version
void cuda_transpose_unpad_c2c_3d(
	int dim_x, int dim_y, int dim_z, // input size
	int red_x, // red_x <= dim_x
	const float *in,   // size: dim_x * dim_y * dim_z
	      float *out)  // size: dim_z * red_x * dim_y (cut in x-direction [of input] from dim_x->red_x)
{
	if (use_naive_kernel) {
		dim3 threads_per_block(8, 8, 8);
		dim3 num_blocks((dim_z+7)/8+1, (red_x+7)/8+1, (dim_y+7/8+1));

		const int logical_gridDim_y = num_blocks.y;
		num_blocks.y *= num_blocks.z;
		num_blocks.z = 1;

		kernel_transpose_unpad_c2c_3d_naive<<<num_blocks, threads_per_block>>>(
			dim_x, dim_y, dim_z,
			red_x,
			in, out,
			logical_gridDim_y
		);		
		checkCudaLastError("kernel_transpose_zeropad_c2c_2d execution failed");
	} else {
		assert(0);
	}

	CUDA_THREAD_SYNCHRONIZE();
}
