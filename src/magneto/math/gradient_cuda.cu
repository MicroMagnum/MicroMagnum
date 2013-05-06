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

#include "config.h"
#include "gradient_cuda.h"

#include <iostream>
using namespace std;

static const int BLOCK_3D_SIZE_X = 8;
static const int BLOCK_3D_SIZE_Y = 8;
static const int BLOCK_3D_SIZE_Z = 8;

__global__ void kernel_gradient_naive(
	int dim_x, int dim_y, int dim_z,
	float delta_x, float delta_y, float delta_z,
	const float *phi, // in: size (dim_x+1) X (dim_y+1) X (dim_z+1)
	float *grad_x, float *grad_y, float *grad_z, // out: size dim_x X dim_y X dim_z
	int logical_grid_dim_y)
{
	// Cell index
	const int x =  blockIdx.x                       * BLOCK_3D_SIZE_X + threadIdx.x;
	const int y = (blockIdx.y % logical_grid_dim_y) * BLOCK_3D_SIZE_Y + threadIdx.y;
	const int z = (blockIdx.y / logical_grid_dim_y) * BLOCK_3D_SIZE_Z + threadIdx.z;

	if (x < dim_x && y < dim_y && z < dim_z) {
		const int phi_sx = 1;
		const int phi_sy = (dim_x+1);
		const int phi_sz = (dim_x+1)*(dim_y+1);
		const int i = phi_sx*x + phi_sy*y + phi_sz*z;

		const double dx = (+ phi[i+1*phi_sx+0*phi_sy+0*phi_sz] - phi[i+0*phi_sx+0*phi_sy+0*phi_sz]
				   + phi[i+1*phi_sx+1*phi_sy+0*phi_sz] - phi[i+0*phi_sx+1*phi_sy+0*phi_sz]
				   + phi[i+1*phi_sx+0*phi_sy+1*phi_sz] - phi[i+0*phi_sx+0*phi_sy+1*phi_sz]
				   + phi[i+1*phi_sx+1*phi_sy+1*phi_sz] - phi[i+0*phi_sx+1*phi_sy+1*phi_sz]) / (4.0 * delta_x);

		const double dy = (+ phi[i+0*phi_sx+1*phi_sy+0*phi_sz] - phi[i+0*phi_sx+0*phi_sy+0*phi_sz]
				   + phi[i+1*phi_sx+1*phi_sy+0*phi_sz] - phi[i+1*phi_sx+0*phi_sy+0*phi_sz]
				   + phi[i+0*phi_sx+1*phi_sy+1*phi_sz] - phi[i+0*phi_sx+0*phi_sy+1*phi_sz]
				   + phi[i+1*phi_sx+1*phi_sy+1*phi_sz] - phi[i+1*phi_sx+0*phi_sy+1*phi_sz]) / (4.0 * delta_y);

		const double dz = (+ phi[i+0*phi_sx+0*phi_sy+1*phi_sz] - phi[i+0*phi_sx+0*phi_sy+0*phi_sz]
				   + phi[i+1*phi_sx+0*phi_sy+1*phi_sz] - phi[i+1*phi_sx+0*phi_sy+0*phi_sz]
				   + phi[i+0*phi_sx+1*phi_sy+1*phi_sz] - phi[i+0*phi_sx+1*phi_sy+0*phi_sz]
				   + phi[i+1*phi_sx+1*phi_sy+1*phi_sz] - phi[i+1*phi_sx+1*phi_sy+0*phi_sz]) / (4.0 * delta_z);

		const int grad_sx = 1;
		const int grad_sy = dim_x;
		const int grad_sz = dim_x * dim_y;
		const int j = grad_sx*x + grad_sy*y + grad_sz*z;
		grad_x[j] = dx;
		grad_y[j] = dy;
		grad_z[j] = dz;
	}
}

void gradient_cuda(double delta_x, double delta_y, double delta_z, const float *phi, VectorMatrix &field)
{
	const int dim_x = field.dimX();
	const int dim_y = field.dimY();
	const int dim_z = field.dimZ();

	dim3 block_dim(BLOCK_3D_SIZE_X, BLOCK_3D_SIZE_Y, BLOCK_3D_SIZE_Z);
	dim3 grid_dim(
		(dim_x + BLOCK_3D_SIZE_X - 1) / BLOCK_3D_SIZE_X, 
		(dim_y + BLOCK_3D_SIZE_Y - 1) / BLOCK_3D_SIZE_Y,
		(dim_z + BLOCK_3D_SIZE_Z - 1) / BLOCK_3D_SIZE_Z
	);

	VectorMatrix::cuda_accessor field_acc(field);
	float *grad_x = field_acc.ptr_x();
	float *grad_y = field_acc.ptr_y();
	float *grad_z = field_acc.ptr_z();

	// Only 2-dimensional grids are supported, so ...
	const int logical_grid_dim_y = grid_dim.y;
	grid_dim.y *= grid_dim.z;
	grid_dim.z = 1;

	kernel_gradient_naive<<<grid_dim, block_dim>>>(
		dim_x, dim_y, dim_z,
		delta_x, delta_y, delta_z,
		phi,
		grad_x, grad_y, grad_z,
		logical_grid_dim_y
	);
	checkCudaLastError("gradient_cuda(): kernel_gradient_naive execution failed!");
	CUDA_THREAD_SYNCHRONIZE();
}
