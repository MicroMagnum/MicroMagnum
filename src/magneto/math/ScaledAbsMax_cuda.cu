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
#include "ScaledAbsMax_cuda.h"

#include "matrix/device/cuda_tools.h"

#include <cfloat>
#include <stdexcept>
#include <cassert>

template <typename real>
__global__ 
static void kernel_max_reduce(const real *in, real *out, int N)
{
	// blockDim = (256,0,0)
	__shared__ real sh[256];
	real tmp1, tmp2;

	const unsigned int bid = gridDim.x * blockIdx.y + blockIdx.x;
	const unsigned int tid = threadIdx.x;
	const unsigned int i = bid * 256 + tid;

	if (i < N) {
		sh[tid] = in[i];
	} else {
		sh[tid] = 0.0;
	}
	__syncthreads();

	if (tid < 128) {
		tmp1 = sh[tid]; tmp2 = sh[tid+128]; sh[tid] = tmp1 > tmp2 ? tmp1 : tmp2;
	} 
	__syncthreads();

	if (tid < 64) {
		tmp1 = sh[tid]; tmp2 = sh[tid+64]; sh[tid] = tmp1 > tmp2 ? tmp1 : tmp2;
	}
	__syncthreads();

	if (tid < 32) {
		volatile real *smem = sh;
		tmp1 = smem[tid]; tmp2 = smem[tid+32]; smem[tid] = tmp1 > tmp2 ? tmp1 : tmp2;
		tmp1 = smem[tid]; tmp2 = smem[tid+16]; smem[tid] = tmp1 > tmp2 ? tmp1 : tmp2;
		tmp1 = smem[tid]; tmp2 = smem[tid+ 8]; smem[tid] = tmp1 > tmp2 ? tmp1 : tmp2;
		tmp1 = smem[tid]; tmp2 = smem[tid+ 4]; smem[tid] = tmp1 > tmp2 ? tmp1 : tmp2;
		tmp1 = smem[tid]; tmp2 = smem[tid+ 2]; smem[tid] = tmp1 > tmp2 ? tmp1 : tmp2;
		tmp1 = smem[tid]; tmp2 = smem[tid+ 1]; smem[tid] = tmp1 > tmp2 ? tmp1 : tmp2;
	}

	// write result for this block to global mem
	if (tid == 0) {
		out[bid] = sh[0];
	}
}

template <typename real>
__global__ 
static void kernel_max_reduce_scaled_sqrlen( // returns maximum of all N 3-vectors in[n].
	const real *in_x, const real *in_y, const real *in_z, const real *scale,
	real *out, 
	int N) 
{
	// blockDim = (256,0,0)
	__shared__ real sh[256];
	real tmp1, tmp2;

	const unsigned int bid = gridDim.x * blockIdx.y + blockIdx.x;
	const unsigned int tid = threadIdx.x;
	const unsigned int i = bid * 256 + tid;

	if (i < N) {
		const real x = in_x[i], y = in_y[i], z = in_z[i];
		const real s = scale[i];
		const real sqrlen = x*x + y*y + z*z;
		if (s == 0.0) {
			sh[tid] = 0.0;
		} else {
			sh[tid] = sqrlen / (s*s);
		}
	} else {
		sh[tid] = 0.0;
	}
	__syncthreads();

	if (tid < 128) { tmp1 = sh[tid]; tmp2 = sh[tid+128]; sh[tid] = tmp1 > tmp2 ? tmp1 : tmp2; } 
	__syncthreads();

	if (tid < 64) { tmp1 = sh[tid]; tmp2 = sh[tid+64]; sh[tid] = tmp1 > tmp2 ? tmp1 : tmp2; }
	__syncthreads();

	if (tid < 32) {
		volatile real *smem = sh;
		tmp1 = smem[tid]; tmp2 = smem[tid+32]; smem[tid] = tmp1 > tmp2 ? tmp1 : tmp2;
		tmp1 = smem[tid]; tmp2 = smem[tid+16]; smem[tid] = tmp1 > tmp2 ? tmp1 : tmp2;
		tmp1 = smem[tid]; tmp2 = smem[tid+ 8]; smem[tid] = tmp1 > tmp2 ? tmp1 : tmp2;
		tmp1 = smem[tid]; tmp2 = smem[tid+ 4]; smem[tid] = tmp1 > tmp2 ? tmp1 : tmp2;
		tmp1 = smem[tid]; tmp2 = smem[tid+ 2]; smem[tid] = tmp1 > tmp2 ? tmp1 : tmp2;
		tmp1 = smem[tid]; tmp2 = smem[tid+ 1]; smem[tid] = tmp1 > tmp2 ? tmp1 : tmp2;
	}

	// write result for this block to global mem
	if (tid == 0) {
		out[bid] = sh[0];
	}
}

template <typename real>
static void alloc_reduce_buffers(real **buf1, real **buf2, int N)
{
	if ((N+255)/256 >= 65535) {
		throw std::runtime_error("Fixme: Support more than 65536*256 cells in cuda reduce routines.");
	}

	const int buf1_size = (N         + 255) / 256;
	const int buf2_size = (buf1_size + 255) / 256;

	real *tmp;
	cudaMalloc((void**)&tmp, (buf1_size + buf2_size) * sizeof(real));
	
	*buf1 = tmp + 0;
	*buf2 = tmp + buf1_size;
}

template <typename real>
static void free_reduce_buffers(real **buf1, real **buf2)
{
	cudaFree(*buf1);
	*buf1 = 0;
	*buf2 = 0;
}

template <typename real>
static real download_scalar(const real *ptr)
{
	real f; 
	if (cudaSuccess != cudaMemcpy((void*)&f, (const void*)ptr, sizeof(real), cudaMemcpyDeviceToHost))
		throw std::runtime_error("Reduce operation: Could not download final result from GPU.");
	return f;
}

template <typename real>
double scaled_abs_max_cuda_impl(VectorMatrix &M, Matrix &scale)
{
	assert(M.size() == scale.size());

	typename VectorMatrix_const_cuda_accessor<real>::t M_acc(M);
	typename Matrix_const_cuda_accessor<real>::t scale_acc(scale);

	const real *Mx = M_acc.ptr_x(), *My = M_acc.ptr_y(), *Mz = M_acc.ptr_z();

	int N = M.size();
	real *buf1, *buf2; 
	alloc_reduce_buffers<real>(&buf1, &buf2, N);

	// First iteration.
	{
		const int B = (N+255)/256;
		kernel_max_reduce_scaled_sqrlen<real><<<B, 256>>>(Mx, My, Mz, scale_acc.ptr(), buf1, N);
		N = B;
	}

	// Rest iterations.
	real *in = buf1, *out = buf2;
	while (N > 1) {
		const int B = (N+255)/256;
		kernel_max_reduce<real><<<B, 256>>>(in, out, N);
		N = B; std::swap(in, out);
	}

	checkCudaLastError("kernel_sum_reduce() execution failed");

	const real squared_result = download_scalar<real>(in); // result is stored at in[0]
	CUDA_THREAD_SYNCHRONIZE();
	free_reduce_buffers<real>(&buf1, &buf2);
	return std::sqrt(static_cast<double>(squared_result));
}

double scaled_abs_max_cuda(VectorMatrix &M, Matrix &scale, bool cuda64)
{
#ifdef HAVE_CUDA_64
	if (cuda64)
	return scaled_abs_max_cuda_impl<double>(M, scale);
	else
#endif
	return scaled_abs_max_cuda_impl<float>(M, scale);
}
