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
#include "llge_cuda.h"

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
__global__ static void kernel_llge(
	real *dM_x, real *dM_y, real *dM_z,
	const real *mag_x, const real *mag_y, const real *mag_z,
	const real *h_eff_x, const real *h_eff_y, const real *h_eff_z,
	const real *precession_factors, const real *damping_factors,
	int num_nodes)
{
	const int     tid = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	const int threadN = __mul24(blockDim.x, gridDim.x);

	for (int i=tid; i<num_nodes; i+=threadN) {
		const real p1 = precession_factors[i];
		const real p2 = damping_factors[i];

		real Ux, Uy, Uz;
		real Vx, Vy, Vz;
		real Mx, My, Mz; Mx =   mag_x[i]; My =   mag_y[i]; Mz =   mag_z[i];
		real Hx, Hy, Hz; Hx = h_eff_x[i]; Hy = h_eff_y[i]; Hz = h_eff_z[i];

		cross3<real>(Ux, Uy, Uz, Mx, My, Mz, Hx, Hy, Hz);
		cross3<real>(Vx, Vy, Vz, Mx, My, Mz, Ux, Uy, Uz);

		dM_x[i] = p1*Ux + p2*Vx;
		dM_y[i] = p1*Uy + p2*Vy;
		dM_z[i] = p1*Uz + p2*Vz;
	}
}

template <typename real>
double llge_cuda_impl(
	const Matrix &f1, const Matrix &f2,
	const VectorMatrix &M,
	const VectorMatrix &H,
	VectorMatrix &dM)
{
	typename VectorMatrix_cuda_accessor<real>::t dM_acc(dM);
	typename VectorMatrix_const_cuda_accessor<real>::t M_acc(M), H_acc(H);
	typename Matrix_const_cuda_accessor<real>::t f1_acc(f1), f2_acc(f2);
	
	kernel_llge<real><<<32, 128>>>(
		dM_acc.ptr_x(), dM_acc.ptr_y(), dM_acc.ptr_z(),
		M_acc.ptr_x(), M_acc.ptr_y(), M_acc.ptr_z(),
		H_acc.ptr_x(), H_acc.ptr_y(), H_acc.ptr_z(),
		f1_acc.ptr(), f2_acc.ptr(), 
		M.size()
	);
	checkCudaLastError("kernel_calculate_llg() execution failed");
	CUDA_THREAD_SYNCHRONIZE();
	return 0.0;
}

double llge_cu32(
	const Matrix &f1, const Matrix &f2,
	const VectorMatrix &M,
	const VectorMatrix &H,
	VectorMatrix &dM)
{
	return llge_cuda_impl<float>(f1, f2, M, H, dM);
}

#ifdef HAVE_CUDA_64
double llge_cu64(
	const Matrix &f1, const Matrix &f2,
	const VectorMatrix &M,
	const VectorMatrix &H,
	VectorMatrix &dM)
{
	return llge_cuda_impl<double>(f1, f2, M, H, dM);
}
#endif
