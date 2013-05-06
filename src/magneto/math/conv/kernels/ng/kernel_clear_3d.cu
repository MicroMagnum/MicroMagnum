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

__global__
void kernel_clear_3d(
	const int dim_x, const int dim_y, const int dim_z, 
	float *out, int out_stride_y, int out_stride_z,
	const int gridDim_y, const int gridDim_z
)
{
	const int gridDim_x = gridDim.x;

	const int blockIdx_x = blockIdx.x;
	const int blockIdx_y = blockIdx.y % gridDim_y;
	const int blockIdx_z = blockIdx.y / gridDim_y; // assert(gridDim.y == gridDim_y*gridDim_z)

	const int x = threadIdx.x + 16 * blockIdx_x;
	const int y = threadIdx.y + 8 * blockIdx_y;
	const int z = threadIdx.z + 8 * blockIdx_z;

	const bool at_border = (blockIdx_x == gridDim_x-1) || (blockIdx_y == gridDim_y-1) || (blockIdx_z == gridDim_z-1);
	//const bool at_border = true;
	if (!at_border) {
#if 0
		const int out_stride_x = 1;
		out[2*(x*out_stride_x+y*out_stride_y+z*out_stride_z)+0] = 0.0f; // clear real part
		out[2*(x*out_stride_x+y*out_stride_y+z*out_stride_z)+1] = 0.0f; // clear imag part
#elif 0
		// (1)
		out[(2*x+0) + 2*(y*out_stride_y+z*out_stride_z)] = 0.0f;
		out[(2*x+1) + 2*(y*out_stride_y+z*out_stride_z)] = 0.0f;
#elif 1
		// (2) Final
		out += 2*(y*out_stride_y+z*out_stride_z);
		//out[2*x+0] = 0.0f;
		//out[2*x+1] = 0.0f;
		out[x+0] = 0.0f;
		out[x+16] = 0.0f;
#else
#		error "Need to select implementation!"
#endif
	} else {
		if (x < dim_x && y < dim_y && z < dim_z) {
			out += 2*(x+y*out_stride_y+z*out_stride_z);
			out[0] = 0.0f;
			out[1] = 0.0f;
		}
	}
}

