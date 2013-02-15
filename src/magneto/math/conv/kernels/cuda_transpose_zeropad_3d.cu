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
#include <iostream>
using namespace std;

static const bool use_naive_kernel = false;

__global__
void kernel_transpose_zeropad_c2c_3d_naive(
	int dim_x, int dim_y, int dim_z,
	int exp_y,
	const float *in, float *out,
	int logical_gridDim_y)
{
	const int in_stride_x = 2 * 1;
	const int in_stride_y = 2 * dim_x;
	const int in_stride_z = 2 * dim_x * dim_y;

	const int out_stride_x = 2 * 1;
	const int out_stride_y = 2 * exp_y;
	const int out_stride_z = 2 * exp_y * dim_z;

	const int blockIdx_x = blockIdx.x;
	const int blockIdx_y = blockIdx.y % logical_gridDim_y;
	const int blockIdx_z = blockIdx.y / logical_gridDim_y;

	const int x = blockIdx_x*8 + threadIdx.x;
	const int y = blockIdx_y*8 + threadIdx.y;
	const int z = blockIdx_z*8 + threadIdx.z;

	if (z < dim_x && y < dim_z && x < exp_y) {
		out += z*out_stride_z + y*out_stride_y + x*out_stride_x;
		if (x < dim_y) { // x in [0..(dim_y-1)]: Transpose
			in += z*in_stride_x + y*in_stride_z + x*in_stride_y;
			out[0] = in[0];
			out[1] = in[1];
		} else { // x in [dim_y..(exp_y-1)]: Zero-pad
			out[0] = 0.0;
			out[1] = 0.0;
		}
	}
}

__global__
void kernel_transpose_zeropad_c2c_3d(
	int dim_x, int dim_y, int dim_z,
	int exp_y,
	const float *in, float *out,

	int logical_gridDim_x,
	int logical_gridDim_y,
	int logical_gridDim_xy,

	const int in_stride_y,
	const int in_stride_z,
	const int out_stride_y,
	const int out_stride_z)
{
	const int lin_block_idx = blockIdx.x + gridDim.x * blockIdx.y;
	const int base_x = 8 * ((lin_block_idx % logical_gridDim_x ));
	const int base_y = 8 * ((lin_block_idx % logical_gridDim_xy) / logical_gridDim_x);
	const int base_z = 8 * ((lin_block_idx                     ) / logical_gridDim_xy);

	__shared__ float shared[8][8+1][2*8+1];

	const int in_stride_x = 2;
	const int out_stride_x = 2;

	// load tile into shared memory
	const int src_x = base_z + threadIdx.x/2;
	const int src_y = base_x + threadIdx.z;
	const int src_z = base_y + threadIdx.y;
	if (src_y < dim_y && src_z < dim_z) {
		in += base_z*in_stride_x + src_z*in_stride_z + src_y*in_stride_y;
		const int imag = threadIdx.x & 1;
		if (src_x < dim_x) {
			shared[threadIdx.x/2  ][threadIdx.y][2*threadIdx.z+imag] = in[threadIdx.x+0];
		}
		if (src_x+4 < dim_x) {
			shared[threadIdx.x/2+4][threadIdx.y][2*threadIdx.z+imag] = in[threadIdx.x+8];
		}
	}

	__syncthreads();

	const int dst_x = base_x + threadIdx.x/2;
	const int dst_y = base_y + threadIdx.y;
	const int dst_z = base_z + threadIdx.z;
	if (dst_z < dim_x && dst_y < dim_z) {
		out += dst_z*out_stride_z + dst_y*out_stride_y + base_x*out_stride_x;
		if (dst_x < exp_y) {
			float tmp = 0.0f;
			if (dst_x < dim_y) {
				tmp = shared[threadIdx.z][threadIdx.y][threadIdx.x+0];
		 	}
			out[threadIdx.x+0] = tmp;
		}
		if (dst_x+4 < exp_y) {
			float tmp = 0.0f;
			if (dst_x+4 < dim_y) {
				tmp = shared[threadIdx.z][threadIdx.y][threadIdx.x+8];
			}
			out[threadIdx.x+8] = tmp;
		}
	}
}

__global__
void kernel_transpose_zeropad_c2c_3d_test(
	int dim_x, int dim_y, int dim_z,
	int exp_y,
	const float *in, float *out,

	int logical_gridDim_x,
	int logical_gridDim_y,
	int logical_gridDim_xy,

	const int in_stride_y,
	const int in_stride_z,
	const int out_stride_y,
	const int out_stride_z)
{
	const int lin_block_idx = blockIdx.x + gridDim.x * blockIdx.y;
	const int base_x = 8 * ((lin_block_idx % logical_gridDim_x ));
	const int base_y = 8 * ((lin_block_idx % logical_gridDim_xy) / logical_gridDim_x);
	const int base_z = 8 * ((lin_block_idx                     ) / logical_gridDim_xy);

	__shared__ float shared[8][8+1][2*8+1];

	const int in_stride_x = 2;
	const int out_stride_x = 2;

	const int src_x = base_z + threadIdx.x/2;
	const int src_y = base_x + threadIdx.z;
	const int src_z = base_y + threadIdx.y;

	const int dst_x = base_x + threadIdx.x/2;
	const int dst_y = base_y + threadIdx.y;
	const int dst_z = base_z + threadIdx.z;

	in  += base_z* in_stride_x + src_z* in_stride_z +  src_y* in_stride_y;
	out +=  dst_z*out_stride_z + dst_y*out_stride_y + base_x*out_stride_x;

	const bool full_block_xz = (base_z+7) < dim_x && (base_y+7) < dim_z;
	const bool full_block    = (base_x+7) < dim_y && full_block_xz;

	if (full_block) {
		const int imag = threadIdx.x & 1;
		shared[threadIdx.x/2  ][threadIdx.y][2*threadIdx.z+imag] = in[threadIdx.x+0];
		shared[threadIdx.x/2+4][threadIdx.y][2*threadIdx.z+imag] = in[threadIdx.x+8];

		__syncthreads();

		out[threadIdx.x+0] = shared[threadIdx.z][threadIdx.y][threadIdx.x+0];
		out[threadIdx.x+8] = shared[threadIdx.z][threadIdx.y][threadIdx.x+8];
	} else {
		const bool zero_block = full_block_xz && (base_x >= dim_y) && (base_x+7 < exp_y);
		if (zero_block) {
			out[threadIdx.x+0] = 0.0f;
			out[threadIdx.x+8] = 0.0f;
		} else {
			// Most generic version. Handles all types of overlaps.

			if (src_y < dim_y && src_z < dim_z) {
				const int imag = threadIdx.x & 1;
				if (src_x < dim_x) {
					shared[threadIdx.x/2  ][threadIdx.y][2*threadIdx.z+imag] = in[threadIdx.x+0];
				}
				if (src_x+4 < dim_x) {
					shared[threadIdx.x/2+4][threadIdx.y][2*threadIdx.z+imag] = in[threadIdx.x+8];
				}
			}

			__syncthreads();

			if (dst_z < dim_x && dst_y < dim_z) {
				if (dst_x < exp_y) {
					float tmp = 0.0f;
					if (dst_x < dim_y) {
						tmp = shared[threadIdx.z][threadIdx.y][threadIdx.x+0];
					}
					out[threadIdx.x+0] = tmp;
				}
				if (dst_x+4 < exp_y) {
					float tmp = 0.0f;
					if (dst_x+4 < dim_y) {
						tmp = shared[threadIdx.z][threadIdx.y][threadIdx.x+8];
					}
					out[threadIdx.x+8] = tmp;
				}
			}
		}
	}
}

void cuda_transpose_zeropad_c2c_3d(
	int dim_x, int dim_y, int dim_z, // input size
	int exp_y, // exp_y >= dim_y
	const float *in_x, const float *in_y, const float *in_z,  // size: dim_x * dim_y
	      float *out_x,      float *out_y,      float *out_z) // size: exp_y * dim_x (zero-padded in x-direction from dim_y -> exp_y)
{
	const float  *in[3] = { in_x,  in_y,  in_z};
	      float *out[3] = {out_x, out_y, out_z};

	if (use_naive_kernel) {
		for (int c=0; c<3; ++c) {
			dim3 threads_per_block(8, 8, 8);
			dim3 num_blocks((exp_y-1)/8+1, (dim_z-1)/8+1, (dim_x-1)/8+1);

			const int logical_gridDim_y = num_blocks.y;
			num_blocks.y *= num_blocks.z;
			num_blocks.z = 1;

			kernel_transpose_zeropad_c2c_3d_naive<<<num_blocks, threads_per_block>>>(
				dim_x, dim_y, dim_z,
				exp_y,
				in[c], out[c],
				logical_gridDim_y
			);		
			checkCudaLastError("kernel_transpose_zeropad_c2c_3d_naive execution failed");
		}
	} else {
		dim3 threads_per_block(8, 8, 8);
		dim3 num_blocks((exp_y-1)/8+1, (dim_z-1)/8+1, (dim_x-1)/8+1);

		const int logical_gridDim_x  = num_blocks.x;
		const int logical_gridDim_y  = num_blocks.y;
		const int logical_gridDim_xy = num_blocks.x * num_blocks.y;

		// TODO: Hmm...
		num_blocks.y *= num_blocks.z;
		num_blocks.z = 1;
		while (num_blocks.y >= 65536) {
			num_blocks.y /= 2;
			num_blocks.x *= 2;
		}

		const int  in_stride_x = 2 * 1;
		const int  in_stride_y = 2 * dim_x;
		const int  in_stride_z = 2 * dim_x * dim_y;
		const int out_stride_x = 2 * 1;
		const int out_stride_y = 2 * exp_y;
		const int out_stride_z = 2 * exp_y * dim_z;

		/*cudaStream_t stream;
		cudaStreamCreate(&stream);*/

		for (int c=0; c<3; ++c) {
			//kernel_transpose_zeropad_c2c_3d<<<num_blocks, threads_per_block>>>(
			kernel_transpose_zeropad_c2c_3d_test<<<num_blocks, threads_per_block>>>(
				dim_x, dim_y, dim_z,
				exp_y,
				in[c], out[c],
				logical_gridDim_x,
				logical_gridDim_y,
				logical_gridDim_xy,
				in_stride_y,
				in_stride_z,
				out_stride_y,
				out_stride_z
			);
			checkCudaLastError("kernel_transpose_zeropad_c2c_3d execution failed");
		}
	}

	CUDA_THREAD_SYNCHRONIZE();
}
