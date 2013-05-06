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

#include "cuda_transpose_unpad.h"
#include "cuda_transpose_unpad_2d.h"
#include "cuda_transpose_unpad_3d.h"

#include <iostream>
using namespace std;

void cuda_transpose_unpad_c2c(
	int dim_x, int dim_y, int dim_z, // input size
	int red_x, // red_x <= dim_x
	const float *in_x, const float *in_y, const float *in_z, // size: dim_x * dim_y * dim_z
	      float *out_x, float *out_y, float *out_z)          // size: dim_z * red_x * dim_y (cut in x-direction [of input] from dim_x->red_x)
{
	const bool is_2d = (dim_y == 1);
	if (!is_2d) {
		cuda_transpose_unpad_c2c_3d(dim_x, dim_y, dim_z, red_x, in_x, in_y, in_z, out_x, out_y, out_z);
	} else {
		cuda_transpose_unpad_c2c_2d(dim_x,        dim_z, red_x, in_x, in_y, in_z, out_x, out_y, out_z);
	}
}

void cuda_transpose_unpad_c2c(
	int dim_x, int dim_y, int dim_z, // input size
	int red_x, // red_x <= dim_x
	const float *in,   // size: dim_x * dim_y * dim_z
	      float *out)  // size: dim_z * red_x * dim_y (cut in x-direction [of input] from dim_x->red_x)
{
	const bool is_2d = (dim_y == 1);
	if (!is_2d) {
		cuda_transpose_unpad_c2c_3d(dim_x, dim_y, dim_z, red_x, in, out);
	} else {
		cuda_transpose_unpad_c2c_2d(dim_x,        dim_z, red_x, in, out);
	}
}

