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

#include "cuda_transpose_zeropad_2d.h"

#include <cuda.h>
#include "matrix/device/cuda_tools.h"

#include <cassert>
//#include <iostream>
//using namespace std;

static const int BLOCK_DIM = 16;

static const bool use_naive_kernel = false;

// This naive transpose kernel suffers from completely non-coalesced writes.
// It can be up to 10x slower than the kernel above for large matrices.
__global__
void kernel_transpose_unpad_c2c_naive(
	int dim_x, int dim_y,
	int red_x,
	const float *in, 
	      float *out)
{
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;

	// (x,y) run through all output matrix cell positions
	if (x < dim_y && y < red_x) {
		const int  in_idx = 2*(y + dim_x * x);
		const int out_idx = 2*(x + dim_y * y);
		out[out_idx+0] = in[in_idx+0];
		out[out_idx+1] = in[in_idx+1];
	}
}

__global__
void kernel_transpose_unpad_c2c(
	int dim_x, int dim_y,
	int red_x,
	const float *in,
	      float *out)
{
	__shared__ float sh[BLOCK_DIM][BLOCK_DIM*2];

	const int base_x = BLOCK_DIM * blockIdx.x;
	const int base_y = BLOCK_DIM * blockIdx.y;

	const int src_pos_x = base_y+threadIdx.x/2;
	const int src_pos_y = base_x+threadIdx.y;
	if (src_pos_y < dim_y) {
		//     | point to tile @ (0,0) |   |  add row offset  | 
		in  += 2*(base_y + dim_x*base_x) + 2*dim_x*threadIdx.y;

		// copy in-tile to shared mem
		if (src_pos_x             < dim_x) {
			sh[threadIdx.y][threadIdx.x          ] = in[threadIdx.x          ]; // columns 0-7  of row 'threadIdx.y'
		}
		if (src_pos_x+BLOCK_DIM/2 < dim_x) {
			sh[threadIdx.y][threadIdx.x+BLOCK_DIM] = in[threadIdx.x+BLOCK_DIM]; // columns 8-15 of row 'threadIdx.y'
		}
	}

	__syncthreads();

	const int dst_pos_x = base_x+threadIdx.x/2;
	const int dst_pos_y = base_y+threadIdx.y;
	if (dst_pos_y < red_x) {
		//     | point to tile @ (0,0) |   |  add row offset  | 
		out += 2*(base_x + dim_y*base_y) + 2*dim_y*threadIdx.y;

		// copy shared-tile to out-tile in transposed order
		const int imag = threadIdx.x & 1;
		if (dst_pos_x             < dim_y) {
			out[threadIdx.x+        0] = sh[threadIdx.x/2            ][2*threadIdx.y+imag];
		}
		if (dst_pos_x+BLOCK_DIM/2 < dim_y) {
			out[threadIdx.x+BLOCK_DIM] = sh[threadIdx.x/2+BLOCK_DIM/2][2*threadIdx.y+imag];
		}
	}
}

void cuda_transpose_unpad_c2c_2d(
	int dim_x, int dim_y,
	int red_x,
	const float * in_x, const float * in_y, const float * in_z, // size (each): dim_x * dim_y (complex)
	      float *out_x,       float *out_y,       float *out_z) // size (each): dim_y * red_x (complex)
{
	// Setup grid and block sizes
	const dim3 grid((dim_y-1)/BLOCK_DIM+1, (red_x-1)/BLOCK_DIM+1, 1);
	const dim3 threads(BLOCK_DIM, BLOCK_DIM, 1);

	const float  *in[3] = { in_x,  in_y,  in_z};
	      float *out[3] = {out_x, out_y, out_z};
	
	for (int c=0; c<3; ++c) {
		if (use_naive_kernel) {
			kernel_transpose_unpad_c2c_naive<<<grid, threads>>>(
				dim_x, dim_y,
				red_x,
				in[c], out[c]
			);
			checkCudaLastError("kernel_transpose_zeropad_c2c_naive execution failed");
		} else {
			kernel_transpose_unpad_c2c<<<grid, threads>>>(
				dim_x, dim_y,
				red_x,
				in[c], out[c]
			);
			checkCudaLastError("kernel_transpose_zeropad_c2c execution failed");
		}
	}
	
	CUDA_THREAD_SYNCHRONIZE();
}

void cuda_transpose_unpad_c2c_2d(
	int dim_x, int dim_y, // input size
	int red_x, // red_x <= dim_x
	const float *in,   // size: dim_x * dim_y
	      float *out)  // size: dim_y * red_x
{
	// Setup grid and block sizes
	const dim3 grid((dim_y-1)/BLOCK_DIM+1, (red_x-1)/BLOCK_DIM+1, 1);
	const dim3 threads(BLOCK_DIM, BLOCK_DIM, 1);

	if (use_naive_kernel) {
		kernel_transpose_unpad_c2c_naive<<<grid, threads>>>(
			dim_x, dim_y,
			red_x,
			in, out
		);
		checkCudaLastError("kernel_transpose_zeropad_c2c_naive execution failed");
	} else {
		kernel_transpose_unpad_c2c<<<grid, threads>>>(
			dim_x, dim_y,
			red_x,
			in, out
		);
		checkCudaLastError("kernel_transpose_zeropad_c2c execution failed");
	}
	
	CUDA_THREAD_SYNCHRONIZE();
}
