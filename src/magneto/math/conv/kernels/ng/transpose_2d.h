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

#ifndef NG_CUDA_TRANSPOSE_2D_H
#define NG_CUDA_TRANSPOSE_2D_H

#include <cuda.h>

void ng_cuda_transpose_zeropad_c2c_2d(
	cudaStream_t s0, cudaStream_t s1,
	int dim_x, int dim_y, // input size
	int exp_y, // exp_y >= dim_y
	const float * in, // size: dim_x * dim_y
	      float *out  // size: exp_y * dim_x (zero-padded in x-direction from dim_y -> exp_y)
);

void ng_cuda_transpose_unpad_c2c_2d(
	cudaStream_t s0,
	int dim_x, int dim_y, // input size
	int red_x, // red_x <= dim_x
	const float * in, // size: dim_x * dim_y
	      float *out  // size: dim_y * red_x
);

#endif
