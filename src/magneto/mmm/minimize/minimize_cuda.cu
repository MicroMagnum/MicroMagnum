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
#include "minimize_cuda.h"

// Vector cross product: A = B x C
template <typename real>
static __inline__ __device__ void cross3(
	real &Ax, real &Ay, real &Az,
	real  Bx, real  By, real  Bz,
	real  Cx, real  Cy, real  Cz)
{
	Ax = By*Cz - Bz*Cy;
	Ay = Bz*Cx - Bx*Cz;
	Az = Bx*Cy - By*Cx;
}


template <typename real>
__global__ static void kernel_minimize(
	real *M2_x, real *M2_y, real *M2_z,
	const real *M_x, const real *M_y, const real *M_z,
	const real *H_x, const real *H_y, const real *H_z,
	const real *f, const real h,
	int num_nodes)
{
	const int     tid = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	const int threadN = __mul24(blockDim.x, gridDim.x);

	for (int i=tid; i<num_nodes; i+=threadN) {

    // shorthand for M and H
    real Mx, My, Mz; Mx = M_x[i]; My = M_y[i]; Mz = M_z[i];
    real Hx, Hy, Hz; Hx = H_x[i]; Hy = H_y[i]; Hz = H_z[i];

    // MxH
    real MxH_x, MxH_y, MxH_z;
    cross3<real>(MxH_x, MxH_y, MxH_z, Mx, My, Mz, Hx, Hy, Hz);

    const real tau = h * f[i];
    const real N   = 4 + tau*tau * (MxH_x*MxH_x + MxH_y*MxH_y + MxH_z*MxH_z);

    M2_x[i] = (4*Mx + 4*tau * (MxH_y*Mz - MxH_z*My) + tau*tau*Mx * (+ MxH_x*MxH_x - MxH_y*MxH_y - MxH_z*MxH_z) + 2*tau*tau*MxH_x * (MxH_y*My + MxH_z*Mz)) / N;
    M2_y[i] = (4*My + 4*tau * (MxH_z*Mx - MxH_x*Mz) + tau*tau*My * (- MxH_x*MxH_x + MxH_y*MxH_y - MxH_z*MxH_z) + 2*tau*tau*MxH_y * (MxH_z*Mz + MxH_x*Mx)) / N;
    M2_z[i] = (4*Mz + 4*tau * (MxH_x*My - MxH_y*Mx) + tau*tau*Mz * (- MxH_x*MxH_x - MxH_y*MxH_y + MxH_z*MxH_z) + 2*tau*tau*MxH_z * (MxH_x*Mx + MxH_y*My)) / N;
	}
}

template <typename real>
double minimize_cuda_impl(
	const Matrix &f, const real h,
	const VectorMatrix &M,
	const VectorMatrix &H,
	VectorMatrix &M2)
{
	typename VectorMatrix_cuda_accessor<real>::t M2_acc(M2);
	typename VectorMatrix_const_cuda_accessor<real>::t M_acc(M), H_acc(H);
	typename Matrix_const_cuda_accessor<real>::t f_acc(f);
	
	kernel_minimize<real><<<32, 128>>>(
		M2_acc.ptr_x(), M2_acc.ptr_y(), M2_acc.ptr_z(),
		M_acc.ptr_x(), M_acc.ptr_y(), M_acc.ptr_z(),
		H_acc.ptr_x(), H_acc.ptr_y(), H_acc.ptr_z(),
		f_acc.ptr(), h, M.size()
	);
	checkCudaLastError("kernel_calculate_llg() execution failed");
	CUDA_THREAD_SYNCHRONIZE();
	return 0.0;
}

double minimize_cu32(
	const Matrix &f, const float h,
	const VectorMatrix &M,
	const VectorMatrix &H,
	VectorMatrix &M2)
{
	return minimize_cuda_impl<float>(f, h, M, H, M2);
}

#ifdef HAVE_CUDA_64
double minimize_cu64(
	const Matrix &f, const double h,
	const VectorMatrix &M,
	const VectorMatrix &H,
	VectorMatrix &M2)
{
	return minimize_cuda_impl<double>(f, h, M, H, M2);
}
#endif
