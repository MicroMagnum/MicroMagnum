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

#include "cuda_transpose_zeropad.h"
#include "cuda_transpose_zeropad_2d.h"
#include "cuda_transpose_zeropad_3d.h"

#include <iostream>
using namespace std;

void cuda_transpose_zeropad_c2c(
	int dim_x, int dim_y, int dim_z, // input size
	int exp_y,
	const float *in_x, const float *in_y, const float *in_z,
	      float *out_x, float *out_y, float *out_z)
{
	const bool is_2d = (dim_z == 1);
	if (!is_2d) {
		cuda_transpose_zeropad_c2c_3d(dim_x, dim_y, dim_z, exp_y, in_x, in_y, in_z, out_x, out_y, out_z);
	} else {
		cuda_transpose_zeropad_c2c_2d(dim_x, dim_y,        exp_y, in_x, in_y, in_z, out_x, out_y, out_z);
	}
}

