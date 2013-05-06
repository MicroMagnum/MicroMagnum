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
__device__ __inline__ static void read_and_store_3d(
	const float * in, const int  in_stride_y, const int  in_stride_z,
	      float *out, const int out_stride_y, const int out_stride_z)
{
	const int x = threadIdx.x, y = threadIdx.y, z = threadIdx.z;
#if 0
	// Translate 'in', 'out' pointers.
	const int  in_stride_x = 1;
	const int out_stride_x = 1;
	in  += 2*(x* in_stride_x + y* in_stride_y + z* in_stride_z); // src at (x,y,z)
	out += 2*(x*out_stride_x + y*out_stride_y + z*out_stride_z); // dst at (x,y,z)
	// Copy the complex number.
	out[0] = in[0]; // copy real part.
	out[1] = in[1]; // copy imag part.
#elif 0
	// (1) Insert indices, simplify.
	out[2*x + 2*(y*out_stride_y + z*out_stride_z) + 0] = in[2*x + 2*(y*in_stride_y + z*in_stride_z) + 0];
	out[2*x + 2*(y*out_stride_y + z*out_stride_z) + 1] = in[2*x + 2*(y*in_stride_y + z*in_stride_z) + 1];
#elif 0
	// (2) Extract offsets due to y- and z-coordinates.
	in  += 2*(y* in_stride_y + z* in_stride_z); 
	out += 2*(y*out_stride_y + z*out_stride_z);
	out[2*x+0] = in[2*x+0];
	out[2*x+1] = in[2*x+1];
#elif 1
	// (3) Reorder. Reads/Writes are now coalesced.
	in  += 2*(y* in_stride_y + z* in_stride_z); 
	out += 2*(y*out_stride_y + z*out_stride_z);
	out[x+           0] = in[x+           0];
	out[x+block_size_x] = in[x+block_size_x];
#else
#error "Need to select implementation!"
#endif
}

template <int block_size_x>
__device__ __inline__ static void read_and_store_rotated_left_3d( // XYZ --> YZX
	const float * in, const int  in_stride_y, const int  in_stride_z,
	      float *out, const int out_stride_y, const int out_stride_z)
{
	// 1) Read at (x,y,z)
	// 2) Store at (y,z,x)
	const int x = threadIdx.x, y = threadIdx.y, z = threadIdx.z;
#if 0
	// Translate 'in', 'out' pointers.
	const int  in_stride_x = 1;
	const int out_stride_x = 1;
	in  += 2*(x* in_stride_x + y* in_stride_y + z* in_stride_z); // src at (x,y,z)
	out += 2*(y*out_stride_x + z*out_stride_y + x*out_stride_z); // dst at (y,z,x)
	// Copy the complex number.
	out[0] = in[0]; // copy real part.
	out[1] = in[1]; // copy imag part.
#elif 0
	// (1) Insert indices.
	const int  in_stride_x = 1;
	const int out_stride_x = 1;
	out[2*(y*out_stride_x + z*out_stride_y + x*out_stride_z)+0] = in[2*(x*in_stride_x + y*in_stride_y + z*in_stride_z)+0];
	out[2*(y*out_stride_x + z*out_stride_y + x*out_stride_z)+1] = in[2*(x*in_stride_x + y*in_stride_y + z*in_stride_z)+1];
#elif 0
	// (2) Simplify. Extract offsets due to y- and z-coordinate.
	in  += 2*(y*in_stride_y + z*in_stride_z); out += 2*(y + z*out_stride_y);
	out[2*x*out_stride_z+0] = in[2*x+0];
	out[2*x*out_stride_z+1] = in[2*x+1];
#elif 0
	// (3) Extract (2*x+0) and (2*x+1) on left hand side.
	in  += 2*(y*in_stride_y + z*in_stride_z); out += 2*(y + z*out_stride_y);
	out[(2*x+0)*out_stride_z             +0] = in[(2*x+0)];
	out[(2*x+1)*out_stride_z-out_stride_z+1] = in[(2*x+1)];
#elif 1
	// (4) Reorder. Reads are now coalesced. Final version.
	in  += 2*(y*in_stride_y + z*in_stride_z); out += 2*(y + z*out_stride_y);
	const int imag = (x & 1) ? (-out_stride_z+1) : 0;
	out[(x+           0)*out_stride_z + imag] = in[x+           0];
	out[(x+block_size_x)*out_stride_z + imag] = in[x+block_size_x];
#else
#error "Need to select implementation!"
#endif
}

template <int block_size_x>
__device__ __inline__ static void read_and_store_rotated_right_3d( // XYZ --> ZXY
	const float * in, const int  in_stride_y, const int  in_stride_z,
	      float *out, const int out_stride_y, const int out_stride_z)
{
	// 1) Read at (x,y,z)
	// 2) Store at (z,x,y)
	const int x = threadIdx.x, y = threadIdx.y, z = threadIdx.z;
#if 0
	// Translate 'in', 'out' pointers.
	const int  in_stride_x = 1;
	const int out_stride_x = 1;
	in  += 2*(x* in_stride_x + y* in_stride_y + z* in_stride_z); // src at (x,y,z)
	out += 2*(z*out_stride_x + x*out_stride_y + y*out_stride_z); // dst at (z,x,y)
	// Copy the complex number.
	out[0] = in[0]; // copy real part.
	out[1] = in[1]; // copy imag part.
#elif 0
	// (1) Insert indices.
	const int  in_stride_x = 1;
	const int out_stride_x = 1;
	out[2*(z*out_stride_x + x*out_stride_y + y*out_stride_z)+0] = in[2*(x*in_stride_x + y*in_stride_y + z*in_stride_z)+0];
	out[2*(z*out_stride_x + x*out_stride_y + y*out_stride_z)+1] = in[2*(x*in_stride_x + y*in_stride_y + z*in_stride_z)+1];
#elif 0
	// (2) Simplify. Extract offsets due to y- and z-coordinate.
	in  += 2*(y*in_stride_y + z*in_stride_z);
	out += 2*(z + y*out_stride_z);
	out[2*x*out_stride_y+0] = in[2*x + 0];
	out[2*x*out_stride_y+1] = in[2*x + 1];
#elif 0
	// (3) Extract (2*x+0) and (2*x+1) on left hand side.
	in  += 2*(y*in_stride_y + z*in_stride_z);
	out += 2*(z + y*out_stride_z);
	out[(2*x+0)*out_stride_y             +0] = in[(2*x+0)];
	out[(2*x+1)*out_stride_y-out_stride_y+1] = in[(2*x+1)];
#elif 1
	// (4) Reorder. Reads are now coalesced. Final version.
	in  += 2*(y*in_stride_y + z*in_stride_z);
	out += 2*(z + y*out_stride_z);
	const int imag = (x & 1) ? (-out_stride_y+1) : 0;
	out[(x+           0)*out_stride_y + imag] = in[x+           0];
	out[(x+block_size_x)*out_stride_y + imag] = in[x+block_size_x];
#else
#error "Need to select implementation!"
#endif
}

__global__
void kernel_rotate_left_3d( // XYZ --> YZX
	const int dim_x, const int dim_y, const int dim_z,
	const float * in, const int  in_stride_y, const int  in_stride_z,  
	      float *out, const int out_stride_y, const int out_stride_z, 
	const int gridDim_y, const int gridDim_z) 
{
	// Setup shared memory layout
	const int P = 1, Q = 0; // Improves speed penalty due to shared memory bank conflicts.

	const int sh_dim_x = 8;
	const int sh_dim_y = 8;
	const int sh_dim_z = 8;

	const int sh_stride_x = 1;
	const int sh_stride_y = sh_stride_x * (sh_dim_x+P);
	const int sh_stride_z = sh_stride_y * (sh_dim_y+Q);

	__shared__ float sh[2*(sh_dim_x+P)*(sh_dim_y+Q)*sh_dim_z];

	// Move to source and dest tiles. 
	const int blockIdx_y = blockIdx.y % gridDim_y;
	const int blockIdx_z = blockIdx.y / gridDim_y;

	const int base_x = sh_dim_x * blockIdx.x, base_y = sh_dim_y * blockIdx_y, base_z = sh_dim_z * blockIdx_z;
	in  += 2*(base_x +  in_stride_y*base_y +  in_stride_z*base_z); // Source tile origin is at (base_x, base_y, base_z).
	out += 2*(base_y + out_stride_y*base_z + out_stride_z*base_x); // Dest   tile origin is at (base_y, base_z, base_x).

	// Copy tile.
	const bool at_border = (blockIdx.x == gridDim.x-1 || blockIdx_y == gridDim_y-1 || blockIdx_z == gridDim_z-1);
	if (!at_border) {
		read_and_store_rotated_left_3d<sh_dim_x>(in, in_stride_y, in_stride_z, sh, sh_stride_y, sh_stride_z);
		__syncthreads();
		read_and_store_3d<sh_dim_x>(sh, sh_stride_y, sh_stride_z, out, out_stride_y, out_stride_z);
	} else {
		in  += 2*(threadIdx.x +  in_stride_y*threadIdx.y +  in_stride_z*threadIdx.z);
		out += 2*(threadIdx.y + out_stride_y*threadIdx.z + out_stride_z*threadIdx.x);
		if (base_x + threadIdx.x < dim_x && base_y + threadIdx.y < dim_y && base_z + threadIdx.z < dim_z) {
			out[0] = in[0];
			out[1] = in[1];
		}
	}
}

__global__
void kernel_rotate_right_3d( // XYZ --> ZXY
	const int dim_x, const int dim_y, const int dim_z,
	const float * in, const int  in_stride_y, const int  in_stride_z,  
	      float *out, const int out_stride_y, const int out_stride_z,
	const int gridDim_y, const int gridDim_z) 
{
	// Setup shared memory layout
	const int P = 1, Q = 0; // Improves speed penalty due to shared memory bank conflicts.

	const int sh_dim_x = 8;
	const int sh_dim_y = 8;
	const int sh_dim_z = 8;

	const int sh_stride_x = 1;
	const int sh_stride_y = sh_stride_x * (sh_dim_x+P);
	const int sh_stride_z = sh_stride_y * (sh_dim_y+Q);

	__shared__ float sh[2*(sh_dim_x+P)*(sh_dim_y+Q)*sh_dim_z];

	// Move to source and dest tiles.
	const int blockIdx_y = blockIdx.y % gridDim_y;
	const int blockIdx_z = blockIdx.y / gridDim_y;

	const int base_x = sh_dim_x * blockIdx.x, base_y = sh_dim_y * blockIdx_y, base_z = sh_dim_z * blockIdx_z;
	in  += 2*(base_x +  in_stride_y*base_y +  in_stride_z*base_z); // Source tile origin is at (base_x, base_y, base_z).
	out += 2*(base_z + out_stride_y*base_x + out_stride_z*base_y); // Dest   tile origin is at (base_z, base_x, base_y).

	// Copy tile.
	const bool at_border = (blockIdx.x == gridDim.x-1 || blockIdx_y == gridDim_y-1 || blockIdx_z == gridDim_z-1);
	if (!at_border) {
		read_and_store_rotated_right_3d<sh_dim_x>(in, in_stride_y, in_stride_z, sh, sh_stride_y, sh_stride_z);
		__syncthreads();
		read_and_store_3d<sh_dim_x>(sh, sh_stride_y, sh_stride_z, out, out_stride_y, out_stride_z);
	} else {
		in  += 2*(threadIdx.x +  in_stride_y*threadIdx.y +  in_stride_z*threadIdx.z);
		out += 2*(threadIdx.z + out_stride_y*threadIdx.x + out_stride_z*threadIdx.y);
		if (base_x + threadIdx.x < dim_x && base_y + threadIdx.y < dim_y && base_z + threadIdx.z < dim_z) {
			out[0] = in[0];
			out[1] = in[1];
		}
		__syncthreads();
	}
}
