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

#include "cpu_transpose_zeropad.h"

static void cpu_transpose_zeropad_c2c_3d_comp(
	int dim_x, int dim_y, int dim_z, // input size
	int exp_y,
	const double *in,  // size: dim_x * dim_y * dim_z
	      double *out) // size: exp_y * dim_z * dim_x (zero-padded in x-direction from dim_y -> exp_y)
{
	const int  in_stride_x = 2 * 1;
	const int  in_stride_y = 2 * dim_x;
	const int  in_stride_z = 2 * dim_x * dim_y;

	const int out_stride_x = 2 * 1;
	const int out_stride_y = 2 * exp_y;
	const int out_stride_z = 2 * exp_y * dim_z;

	// (x,y,z) loop through the out matrix indices
	for (int z=0; z<dim_x; ++z)
	for (int y=0; y<dim_z; ++y)
	for (int x=0; x<exp_y; ++x) {
		double *dst = out + z*out_stride_z + y*out_stride_y + x*out_stride_x;
		if (x < dim_y) { // x in [0..(dim_y-1)]: Transpose
			const double *src = in + z*in_stride_x + y*in_stride_z + x*in_stride_y;
			dst[0] = src[0];
			dst[1] = src[1];
		} else { // x in [dim_y..(exp_y-1)]: Zero-pad
			dst[0] = 0.0;
			dst[1] = 0.0;
		}
	}
}

static void cpu_transpose_zeropad_c2c_2d_comp(
	int dim_x, int dim_y, // input size
	int exp_y,
	const double *in,  // size: dim_x * dim_y
	      double *out) // size: exp_y * dim_x (zero-padded in x-direction from dim_y -> exp_y)
{
	const int  in_stride_x = 2 * 1;
	const int  in_stride_y = 2 * dim_x;

	const int out_stride_x = 2 * 1;
	const int out_stride_y = 2 * exp_y;

	// (x,y) loop through the out matrix indices
	for (int y=0; y<dim_x; ++y)
	for (int x=0; x<exp_y; ++x) {
		double *dst = out + y*out_stride_y + x*out_stride_x;
		if (x < dim_y) { // x in [0..(dim_y-1)]: Transpose
			const double *src = in + y*in_stride_x + x*in_stride_y;
			dst[0] = src[0];
			dst[1] = src[1];
		} else { // x in [dim_y..(exp_y-1)]: Zero-pad
			dst[0] = 0.0;
			dst[1] = 0.0;
		}
	}
}

void cpu_transpose_zeropad_c2c(
	int dim_x, int dim_y, int dim_z,
	int exp_y,
	const double  *in_x, const double  *in_y, const double  *in_z, 
	      double *out_x,       double *out_y,       double *out_z)          
{
	const bool is_2d = (dim_z == 1);
	if (!is_2d) {
		cpu_transpose_zeropad_c2c_3d_comp(dim_x, dim_y, dim_z, exp_y, in_x, out_x);
		cpu_transpose_zeropad_c2c_3d_comp(dim_x, dim_y, dim_z, exp_y, in_y, out_y);
		cpu_transpose_zeropad_c2c_3d_comp(dim_x, dim_y, dim_z, exp_y, in_z, out_z);
	} else {
		cpu_transpose_zeropad_c2c_2d_comp(dim_x, dim_y,        exp_y, in_x, out_x);
		cpu_transpose_zeropad_c2c_2d_comp(dim_x, dim_y,        exp_y, in_y, out_y);
		cpu_transpose_zeropad_c2c_2d_comp(dim_x, dim_y,        exp_y, in_z, out_z);
	}
}

