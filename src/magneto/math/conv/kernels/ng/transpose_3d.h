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

#ifndef NG_CUDA_TRANSPOSE_3D_H
#define NG_CUDA_TRANSPOSE_3D_H

#include <cuda.h>

void ng_cuda_transpose_zeropad_c2c_3d( // xyz -> Yzx
	cudaStream_t s0, cudaStream_t s1,
	int dim_x, int dim_y, int dim_z, // input size
	int exp_y, // exp_y >= dim_y
	const float * in, // size: dim_x * dim_y * dim_z
	      float *out  // size: exp_y * dim_z * dim_x
);

void ng_cuda_transpose_unpad_c2c_3d( // XYZ -> ZxY
	int dim_x, int dim_y, int dim_z, // input size
	int red_x, // red_x <= dim_x
	const float *in,   // size: dim_x * dim_y * dim_z
	      float *out   // size: dim_z * red_x * dim_y (cut in x-direction from dim_x->red_x)
);

#endif
