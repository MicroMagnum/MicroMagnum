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
#include "runge_kutta_cuda.h"

#include <cuda.h>
#include <assert.h>
#include <stdexcept>

#include "Logger.h"

template <typename real>
__global__
void kernel_runge_kutta_combine_result4(
	real h,
	/*in*/
	real c0, real ec0, const real *k0,
	real c1, real ec1, const real *k1,
	real c2, real ec2, const real *k2,
	real c3, real ec3, const real *k3,
	/*inout*/
	real *y,
	real *y_error,
	int N)
{
	const int     tid = blockDim.x * blockIdx.x + threadIdx.x;
	const int threadN = blockDim.x * gridDim.x;

	for (int i=tid; i<N; i+=threadN) {
		const real k0_ = k0[i];
		const real k1_ = k1[i];
		const real k2_ = k2[i];
		const real k3_ = k3[i];

		y      [i] += h * ( c0*k0_ +  c1*k1_ +  c2*k2_ +  c3*k3_);
		y_error[i]  = h * (ec0*k0_ + ec1*k1_ + ec2*k2_ + ec3*k3_);
	}
}

template <typename real>
__global__
void kernel_runge_kutta_combine_result5(
	real h,
	/*in*/
	real c0, real ec0, const real *k0,
	real c1, real ec1, const real *k1,
	real c2, real ec2, const real *k2,
	real c3, real ec3, const real *k3,
	real c4, real ec4, const real *k4,
	/*inout*/
	real *y,
	real *y_error,
	int N)
{
	const int     tid = blockDim.x * blockIdx.x + threadIdx.x;
	const int threadN = blockDim.x * gridDim.x;

	for (int i=tid; i<N; i+=threadN) {
		const real k0_ = k0[i];
		const real k1_ = k1[i];
		const real k2_ = k2[i];
		const real k3_ = k3[i];
		const real k4_ = k4[i];

		y      [i] += h * ( c0*k0_ +  c1*k1_ +  c2*k2_ +  c3*k3_ +  c4*k4_);
		y_error[i]  = h * (ec0*k0_ + ec1*k1_ + ec2*k2_ + ec3*k3_ + ec4*k4_);
	}
}

//////////////////////////////////////////////////////////////////////////////
// WRAPPERS THAT CALL THE KERNELS ABOVE                                     //
//////////////////////////////////////////////////////////////////////////////

void rk_prepare_step_cuda(
	int step, double h, ButcherTableau &tab,
	const VectorMatrix &k0, const VectorMatrix &k1, const VectorMatrix &k2,
	const VectorMatrix &k3, const VectorMatrix &k4, const VectorMatrix &k5,
	const VectorMatrix &y, VectorMatrix &ytmp, bool cuda64)
{
	// TODO: Write custom kernels for performance and better numerical precision!
	ytmp.assign(y);
	switch (step) {
		case 0: break;
		case 1: ytmp.add(k0, h*tab.b[1][0]); break;
		case 2: ytmp.add(k0, h*tab.b[2][0]);
		        ytmp.add(k1, h*tab.b[2][1]); break;
		case 3: ytmp.add(k0, h*tab.b[3][0]);
		        ytmp.add(k1, h*tab.b[3][1]);
		        ytmp.add(k2, h*tab.b[3][2]); break;
		case 4: ytmp.add(k0, h*tab.b[4][0]);
		        ytmp.add(k1, h*tab.b[4][1]);
		        ytmp.add(k2, h*tab.b[4][2]);
		        ytmp.add(k3, h*tab.b[4][3]); break;
		case 5: ytmp.add(k0, h*tab.b[5][0]);
		        ytmp.add(k1, h*tab.b[5][1]);
		        ytmp.add(k2, h*tab.b[5][2]);
		        ytmp.add(k3, h*tab.b[5][3]);
		        ytmp.add(k4, h*tab.b[5][4]); break;
		default: 
			throw std::runtime_error("Cant handle runge-kutta methods with more than 6 steps (not implemented)");
	}
}

template <typename real>
void rk_combine_result_cuda_impl(
	double h, ButcherTableau &tab,
	const VectorMatrix &k0, const VectorMatrix &k1, const VectorMatrix &k2, const VectorMatrix &k3,
	VectorMatrix &y, VectorMatrix &y_error)
{
	typename VectorMatrix_const_cuda_accessor<real>::t k0_acc(k0), k1_acc(k1), k2_acc(k2), k3_acc(k3);
	typename VectorMatrix_cuda_accessor<real>::t y_acc(y), y_error_acc(y_error);

	kernel_runge_kutta_combine_result4<real><<<32, 128>>>(h, tab.c[0], tab.ec[0], k0_acc.ptr_x(), tab.c[1], tab.ec[1], k1_acc.ptr_x(), tab.c[2], tab.ec[2], k2_acc.ptr_x(), tab.c[3], tab.ec[3], k3_acc.ptr_x(), y_acc.ptr_x(), y_error_acc.ptr_x(), y.size());
	kernel_runge_kutta_combine_result4<real><<<32, 128>>>(h, tab.c[0], tab.ec[0], k0_acc.ptr_y(), tab.c[1], tab.ec[1], k1_acc.ptr_y(), tab.c[2], tab.ec[2], k2_acc.ptr_y(), tab.c[3], tab.ec[3], k3_acc.ptr_y(), y_acc.ptr_y(), y_error_acc.ptr_y(), y.size());
	kernel_runge_kutta_combine_result4<real><<<32, 128>>>(h, tab.c[0], tab.ec[0], k0_acc.ptr_z(), tab.c[1], tab.ec[1], k1_acc.ptr_z(), tab.c[2], tab.ec[2], k2_acc.ptr_z(), tab.c[3], tab.ec[3], k3_acc.ptr_z(), y_acc.ptr_z(), y_error_acc.ptr_z(), y.size());

	checkCudaLastError("kernel_runge_kutta_combine_result5() execution failed");
	checkCudaSuccess(cudaThreadSynchronize());
}

template <typename real>
void rk_combine_result_cuda_impl(
	double h, ButcherTableau &tab,
	const VectorMatrix &k0, const VectorMatrix &k1, const VectorMatrix &k2,
	const VectorMatrix &k3, const VectorMatrix &k4, const VectorMatrix &k5,
	VectorMatrix &y, VectorMatrix &y_error)
{
	assert(tab.c[1] == 0.0);
	assert(tab.ec[1] == 0.0);

	typename VectorMatrix_const_cuda_accessor<real>::t k0_acc(k0),             k2_acc(k2);
	typename VectorMatrix_const_cuda_accessor<real>::t k3_acc(k3), k4_acc(k4), k5_acc(k5);
	typename VectorMatrix_cuda_accessor<real>::t y_acc(y), y_error_acc(y_error);

	kernel_runge_kutta_combine_result5<real><<<32, 128>>>(h, tab.c[0], tab.ec[0], k0_acc.ptr_x(), tab.c[2], tab.ec[2], k2_acc.ptr_x(), tab.c[3], tab.ec[3], k3_acc.ptr_x(), tab.c[4], tab.ec[4], k4_acc.ptr_x(), tab.c[5], tab.ec[5], k5_acc.ptr_x(), y_acc.ptr_x(), y_error_acc.ptr_x(), y.size());
	kernel_runge_kutta_combine_result5<real><<<32, 128>>>(h, tab.c[0], tab.ec[0], k0_acc.ptr_y(), tab.c[2], tab.ec[2], k2_acc.ptr_y(), tab.c[3], tab.ec[3], k3_acc.ptr_y(), tab.c[4], tab.ec[4], k4_acc.ptr_y(), tab.c[5], tab.ec[5], k5_acc.ptr_y(), y_acc.ptr_y(), y_error_acc.ptr_y(), y.size());
	kernel_runge_kutta_combine_result5<real><<<32, 128>>>(h, tab.c[0], tab.ec[0], k0_acc.ptr_z(), tab.c[2], tab.ec[2], k2_acc.ptr_z(), tab.c[3], tab.ec[3], k3_acc.ptr_z(), tab.c[4], tab.ec[4], k4_acc.ptr_z(), tab.c[5], tab.ec[5], k5_acc.ptr_z(), y_acc.ptr_z(), y_error_acc.ptr_z(), y.size());

	checkCudaLastError("kernel_runge_kutta_combine_result5() execution failed");
	checkCudaSuccess(cudaThreadSynchronize());
}

void rk_combine_result_cuda(
	double h, ButcherTableau &tab,
	const VectorMatrix &k0, const VectorMatrix &k1, const VectorMatrix &k2, const VectorMatrix &k3,
	VectorMatrix &y, VectorMatrix &y_error,
	bool cuda64)
{
#ifdef HAVE_CUDA_64
	if (cuda64)
	rk_combine_result_cuda_impl<double>(h, tab, k0, k1, k2, k3, y, y_error);
	else
#endif
	rk_combine_result_cuda_impl<float>(h, tab, k0, k1, k2, k3, y, y_error);
}

void rk_combine_result_cuda(
	double h, ButcherTableau &tab,
	const VectorMatrix &k0, const VectorMatrix &k1, const VectorMatrix &k2,
	const VectorMatrix &k3, const VectorMatrix &k4, const VectorMatrix &k5,
	VectorMatrix &y, VectorMatrix &y_error,
	bool cuda64)
{
#ifdef HAVE_CUDA_64
	if (cuda64)
	rk_combine_result_cuda_impl<double>(h, tab, k0, k1, k2, k3, k4, k5, y, y_error);
	else
#endif
	rk_combine_result_cuda_impl<float>(h, tab, k0, k1, k2, k3, k4, k5, y, y_error);
}
