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

#include "cpu_copy_pad.h"

static void cpu_copy_pad_r2c_comp(
	int dim_x, int dim_y, int dim_z,
	int exp_x,
	const double *in,  // size: dim_x * dim_y * dim_z
	      double *out) // size: exp_x * dim_y * dim_z (zero-padded in x-direction from dim_x -> exp_x)
{
	const int  in_stride_x = 1 * 1;
	const int  in_stride_y = 1 * dim_x;
	const int  in_stride_z = 1 * dim_x * dim_y;

	const int out_stride_x = 2 * 1;
	const int out_stride_y = 2 * exp_x;
	const int out_stride_z = 2 * exp_x * dim_y;

	// (x,y,z) loop through the out matrix indices
	for (int z=0; z<dim_z; ++z)
	for (int y=0; y<dim_y; ++y)
	for (int x=0; x<exp_x; ++x) {
		double *dst = out + z*out_stride_z + y*out_stride_y + x*out_stride_x;
		if (x < dim_x) {
			const double *src = in + z*in_stride_z + y*in_stride_y + x*in_stride_x;
			dst[0] = src[0];
			dst[1] = 0.0;
		} else {
			dst[0] = 0.0;
			dst[1] = 0.0;
		}
	}
}

void cpu_copy_pad_r2c(
	int dim_x, int dim_y, int dim_z,
	int exp_x,
	const double *in_x, const double *in_y, const double *in_z,
	      double *out_x,      double *out_y,      double *out_z)
{
	cpu_copy_pad_r2c_comp(dim_x, dim_y, dim_z, exp_x, in_x, out_x);
	cpu_copy_pad_r2c_comp(dim_x, dim_y, dim_z, exp_x, in_y, out_y);
	cpu_copy_pad_r2c_comp(dim_x, dim_y, dim_z, exp_x, in_z, out_z);
}

static void cpu_copy_pad_r2r_comp(
	int dim_x, int dim_y, int dim_z,
	int exp_x,
	const double *in,  // size: dim_x * dim_y * dim_z
	      double *out) // size: exp_x * dim_y * dim_z (zero-padded in x-direction from dim_x -> exp_x)
{
	const int  in_stride_x = 1 * 1;
	const int  in_stride_y = 1 * dim_x;
	const int  in_stride_z = 1 * dim_x * dim_y;

	const int out_stride_x = 1 * 1;
	const int out_stride_y = 1 * exp_x;
	const int out_stride_z = 1 * exp_x * dim_y;

	// (x,y,z) loop through the out matrix indices
	for (int z=0; z<dim_z; ++z)
	for (int y=0; y<dim_y; ++y)
	for (int x=0; x<exp_x; ++x) {
		double *dst = out + z*out_stride_z + y*out_stride_y + x*out_stride_x;
		if (x < dim_x) {
			const double *src = in + z*in_stride_z + y*in_stride_y + x*in_stride_x;
			dst[0] = src[0];
		} else {
			dst[0] = 0.0;
		}
	}
}

void cpu_copy_pad_r2r(
	int dim_x, int dim_y, int dim_z,
	int exp_x,
	const double *in_x, const double *in_y, const double *in_z,
	      double *out_x,      double *out_y,      double *out_z)
{
	cpu_copy_pad_r2r_comp(dim_x, dim_y, dim_z, exp_x, in_x, out_x);
	cpu_copy_pad_r2r_comp(dim_x, dim_y, dim_z, exp_x, in_y, out_y);
	cpu_copy_pad_r2r_comp(dim_x, dim_y, dim_z, exp_x, in_z, out_z);
}

