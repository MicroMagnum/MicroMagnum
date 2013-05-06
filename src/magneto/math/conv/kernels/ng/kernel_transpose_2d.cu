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

#include "kernels.h"

template <int block_size_x>
__device__ __inline__ static void read_and_store_2d(
	const float * in, const int  in_stride_y,
	      float *out, const int out_stride_y)
{
	const int x = threadIdx.x, y = threadIdx.y;
#if 0
	// Translate 'in', 'out' pointers.
	const int  in_stride_x = 1;
	const int out_stride_x = 1;
	in  += 2*(x* in_stride_x + y* in_stride_y); // src at (x,y)
	out += 2*(x*out_stride_x + y*out_stride_y); // dst at (x,y)
	// Copy the complex number.
	out[0] = in[0]; // copy real part.
	out[1] = in[1]; // copy imag part.
#elif 0
	// (1) Insert indices, simplify.
	out[2*x + 2*y*out_stride_y + 0] = in[2*x + 2*y*in_stride_y + 0];
	out[2*x + 2*y*out_stride_y + 1] = in[2*x + 2*y*in_stride_y + 1];
#elif 0
	// (2) Extract offsets by y-coordinate.
	in += 2*y*in_stride_y; out += 2*y*out_stride_y;
	out[2*x+0] = in[2*x+0];
	out[2*x+1] = in[2*x+1];
#elif 1
	// (3) Reorder. Reads/Writes are now coalesced.
	in += 2*y*in_stride_y; out += 2*y*out_stride_y;
	out[x+           0] = in[x+           0];
	out[x+block_size_x] = in[x+block_size_x];
#else
#error "Need to select implementation!"
#endif
}

template <int block_size_x>
__device__ __inline__ static void read_and_store_transposed_2d(
	const float * in, const int  in_stride_y,
	      float *out, const int out_stride_y)
{
	// 1) Read at (x,y)
	// 2) Store at (y,x)
	const int x = threadIdx.x, y = threadIdx.y;
#if 0
	// Translate 'in', 'out' pointers.
	const int  in_stride_x = 1;
	const int out_stride_x = 1;
	in  += 2*(x* in_stride_x + y* in_stride_y); // src at (x,y)
	out += 2*(y*out_stride_x + x*out_stride_y); // dst at (y,x)
	// Copy the complex number.
	out[0] = in[0]; // copy real part.
	out[1] = in[1]; // copy imag part.
#elif 0
	// (1) Insert indices.
	const int  in_stride_x = 1;
	const int out_stride_x = 1;
	out[2*(y*out_stride_x + x*out_stride_y)+0] = in[2*(x* in_stride_x + y* in_stride_y)+0];
	out[2*(y*out_stride_x + x*out_stride_y)+1] = in[2*(x* in_stride_x + y* in_stride_y)+1];
#elif 0
	// (2) Simplify. Extract offsets due to y-coordinate.
	in += 2*y*in_stride_y; out += 2*y; 
	out[2*x*out_stride_y + 0] = in[2*x + 0];
	out[2*x*out_stride_y + 1] = in[2*x + 1];
#elif 0
	// (3) Extract (2*x+0) and (2*x+1) on left hand side.
	in += 2*y*in_stride_y; out += 2*y; 
	out[(2*x+0)*out_stride_y +                  0] = in[(2*x+0)];
	out[(2*x+1)*out_stride_y - (out_stride_y + 1)] = in[(2*x+1)];
#elif 1
	// (4) Reorder. Reads are now coalesced. Final version.
	in += 2*y*in_stride_y; out += 2*y;
	const int imag = (x & 1) ? (-out_stride_y+1) : 0;
	out[(x+           0)*out_stride_y + imag] = in[x+           0];
	out[(x+block_size_x)*out_stride_y + imag] = in[x+block_size_x];
#else
#error "Need to select implementation!"
#endif
}

__global__
void kernel_transpose_2d( // XY -> YX
	const int dim_x, const int dim_y,
	const float * in, const int  in_stride_y,  
	      float *out, const int out_stride_y) 
{
	const int Q = 2; // Improves shared memory bank conflicts.

	__shared__ float sh[2*16*(16+Q)];

	// Move to source and dest tiles. 
	const int base_x = 16 * blockIdx.x, base_y = 16 * blockIdx.y;
	in  += 2*(base_x +  in_stride_y*base_y); // Source tile origin is at (base_x, base_y).
	out += 2*(base_y + out_stride_y*base_x); // Dest   tile origin is at (base_y, base_x).

	const bool at_border = (blockIdx.x == gridDim.x-1 || blockIdx.y == gridDim.y-1);// && (overlap_x > 0 || overlap_y > 0));
	if (!at_border) { // This branch is not that expensive because all threads within a block take the same branch.
		read_and_store_transposed_2d<16>(in, in_stride_y, sh, 16+Q);
		__syncthreads();
		read_and_store_2d<16>(sh, 16+Q, out, out_stride_y);
	} else {
		in  += 2*(threadIdx.x +  in_stride_y*threadIdx.y);
		out += 2*(threadIdx.y + out_stride_y*threadIdx.x);
		if (base_x + threadIdx.x < dim_x && base_y + threadIdx.y < dim_y) {
			out[0] = in[0];
			out[1] = in[1];
		}
		__syncthreads();
	}
}

