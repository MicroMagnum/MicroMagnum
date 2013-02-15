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

#include "cpu_transpose_unpad.h"

static void cpu_transpose_unpad_c2c_3d_comp(
	int dim_x, int dim_y, int dim_z,
	int red_x,
	const double *in,  // size: dim_x * dim_y * dim_z
	      double *out) // size: dim_z * red_x * dim_y (cut in x-direction [of input] from dim_x->red_x)
{
	const int in_stride_x = 2 * 1;
	const int in_stride_y = 2 * dim_x;
	const int in_stride_z = 2 * dim_x * dim_y;

	const int out_stride_x = 2 * 1;
	const int out_stride_y = 2 * dim_z;
	const int out_stride_z = 2 * dim_z * red_x;

	// (x,y,z) loop through the out matrix indices
	for (int z=0; z<dim_y; ++z)
	for (int y=0; y<red_x; ++y)
	for (int x=0; x<dim_z; ++x) {
		      double *dst = out + z*out_stride_z + y*out_stride_y + x*out_stride_x;
		const double *src = in  + z* in_stride_y + y* in_stride_x + x* in_stride_z;
		dst[0] = src[0];
		dst[1] = src[1];
	}
}

static void cpu_transpose_unpad_c2c_2d_comp(
	int dim_x, int dim_z,
	int red_x,
	const double *in,   // size: dim_x * dim_z
	      double *out)  // size: dim_z * red_x (cut in x-direction [of input] from dim_x->red_x)
{
	const int in_stride_x = 2 * 1;
	const int in_stride_y = 2 * dim_x;

	const int out_stride_x = 2 * 1;
	const int out_stride_y = 2 * dim_z;

	// (x,y,z) loop through the out matrix indices
	for (int y=0; y<red_x; ++y)
	for (int x=0; x<dim_z; ++x) {
		      double *dst = out + y*out_stride_y + x*out_stride_x;
		const double *src =  in + y* in_stride_x + x* in_stride_y;
		dst[0] = src[0];
		dst[1] = src[1];
	}
}

void cpu_transpose_unpad_c2c(
	int dim_x, int dim_y, int dim_z, // input size
	int red_x, // red_x <= dim_x
	const double *in_x, const double *in_y, const double *in_z,  // size: dim_x * dim_y * dim_z
	      double *out_x,     double *out_y,      double *out_z)  // size: dim_z * red_x * dim_y (cut in x-direction [of input] from dim_x->red_x)
{
	const bool is_2d = (dim_y == 1);
	if (is_2d) {
		cpu_transpose_unpad_c2c_2d_comp(dim_x,        dim_z, red_x, in_x, out_x);
		cpu_transpose_unpad_c2c_2d_comp(dim_x,        dim_z, red_x, in_y, out_y);
		cpu_transpose_unpad_c2c_2d_comp(dim_x,        dim_z, red_x, in_z, out_z);
	} else {
		cpu_transpose_unpad_c2c_3d_comp(dim_x, dim_y, dim_z, red_x, in_x, out_x);
		cpu_transpose_unpad_c2c_3d_comp(dim_x, dim_y, dim_z, red_x, in_y, out_y);
		cpu_transpose_unpad_c2c_3d_comp(dim_x, dim_y, dim_z, red_x, in_z, out_z);
	}
}

void cpu_transpose_unpad_c2c(
	int dim_x, int dim_y, int dim_z, // input size
	int red_x, // red_x <= dim_x
	const double  *in_x,  // size: dim_x * dim_y * dim_z
	      double *out_x)  // size: dim_z * red_x * dim_y (cut in x-direction [of input] from dim_x->red_x)
{
	const bool is_2d = (dim_y == 1);
	if (is_2d) {
		cpu_transpose_unpad_c2c_2d_comp(dim_x,        dim_z, red_x, in_x, out_x);
	} else {
		cpu_transpose_unpad_c2c_3d_comp(dim_x, dim_y, dim_z, red_x, in_x, out_x);
	}
}
