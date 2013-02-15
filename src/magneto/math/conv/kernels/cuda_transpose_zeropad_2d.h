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

#ifndef TRANSPOSE_ZEROPAD_2D_H
#define TRANSPOSE_ZEROPAD_2D_H

// xyz -> Yzx
void cuda_transpose_zeropad_c2c_2d(
	int dim_x, int dim_y, // input size
	int exp_y, // exp_y >= dim_y
	const float *in_x, const float *in_y, const float *in_z, // size: dim_x * dim_y
	      float *out_x, float *out_y, float *out_z);         // size: exp_y * dim_x (zero-padded in x-direction from dim_y -> exp_y)

#endif
