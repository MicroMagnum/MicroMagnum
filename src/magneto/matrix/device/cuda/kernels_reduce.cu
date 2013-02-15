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
#include "kernels_reduce.h"

#include "matrix/device/cuda_tools.h"

#include <stdexcept>

#include <cfloat> // FLT_MAX, DBL_MAX
template <typename T> __host__ __device__ inline      T float_max() { return 0; }
template <>           __host__ __device__ inline  float float_max() { return FLT_MAX; }
template <>           __host__ __device__ inline double float_max() { return DBL_MAX; }

template <typename T>
__global__ 
static void kernel_min_reduce(const T * __restrict__ in, T * __restrict__ out, int N)
{
	__shared__ T sh[256]; // blockDim = (256,0,0)
	T tmp1, tmp2;

	const unsigned int bid = gridDim.x * blockIdx.y + blockIdx.x;
	const unsigned int tid = threadIdx.x;
	const unsigned int i = bid * 256 + tid;

	if (i < N) {
		sh[tid] = in[i];
	} else {
		sh[tid] = +float_max<T>(); // most positive representable number.
	}
	__syncthreads();

	if (tid < 128) {
		tmp1 = sh[tid]; tmp2 = sh[tid+128]; sh[tid] = tmp1 < tmp2 ? tmp1 : tmp2;
	} 
	__syncthreads();

	if (tid < 64) {
		tmp1 = sh[tid]; tmp2 = sh[tid+64]; sh[tid] = tmp1 < tmp2 ? tmp1 : tmp2;
	}
	__syncthreads();

	if (tid < 32) {
		volatile T *smem = sh;
		tmp1 = smem[tid]; tmp2 = smem[tid+32]; smem[tid] = tmp1 < tmp2 ? tmp1 : tmp2;
		tmp1 = smem[tid]; tmp2 = smem[tid+16]; smem[tid] = tmp1 < tmp2 ? tmp1 : tmp2;
		tmp1 = smem[tid]; tmp2 = smem[tid+ 8]; smem[tid] = tmp1 < tmp2 ? tmp1 : tmp2;
		tmp1 = smem[tid]; tmp2 = smem[tid+ 4]; smem[tid] = tmp1 < tmp2 ? tmp1 : tmp2;
		tmp1 = smem[tid]; tmp2 = smem[tid+ 2]; smem[tid] = tmp1 < tmp2 ? tmp1 : tmp2;
		tmp1 = smem[tid]; tmp2 = smem[tid+ 1]; smem[tid] = tmp1 < tmp2 ? tmp1 : tmp2;
	}

	// write result for this block to global mem
	if (tid == 0) {
		out[bid] = sh[0];
	}
}

template <typename T>
__global__ 
static void kernel_max_reduce(const T *in, T *out, int N)
{
	__shared__ T sh[256]; // blockDim = (256,0,0)
	T tmp1, tmp2;

	const unsigned int bid = gridDim.x * blockIdx.y + blockIdx.x;
	const unsigned int tid = threadIdx.x;
	const unsigned int i = bid * 256 + tid;

	if (i < N) {
		sh[tid] = in[i];
	} else {
		sh[tid] = -float_max<T>(); // most negative representable number.
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
		volatile T *smem = sh;
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

template <typename T>
__global__ 
static void kernel_sum_reduce(const T *in, T *out, int N)
{
	__shared__ T sh[256]; // blockDim = (256,0,0)
	T tmp1, tmp2;

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
		tmp1 = sh[tid]; tmp2 = sh[tid+128]; sh[tid] = tmp1 + tmp2;
	} 
	__syncthreads();

	if (tid < 64) {
		tmp1 = sh[tid]; tmp2 = sh[tid+64]; sh[tid] = tmp1 + tmp2;
	}
	__syncthreads();

	if (tid < 32) {
		volatile T *smem = sh;
		tmp1 = smem[tid]; tmp2 = smem[tid+32]; smem[tid] = tmp1 + tmp2;
		tmp1 = smem[tid]; tmp2 = smem[tid+16]; smem[tid] = tmp1 + tmp2;
		tmp1 = smem[tid]; tmp2 = smem[tid+ 8]; smem[tid] = tmp1 + tmp2;
		tmp1 = smem[tid]; tmp2 = smem[tid+ 4]; smem[tid] = tmp1 + tmp2;
		tmp1 = smem[tid]; tmp2 = smem[tid+ 2]; smem[tid] = tmp1 + tmp2;
		tmp1 = smem[tid]; tmp2 = smem[tid+ 1]; smem[tid] = tmp1 + tmp2;
	}

	// write result for this block to global mem
	if (tid == 0) {
		out[bid] = sh[0];
	}
}

template <typename T>
__global__ 
static void kernel_max_reduce_sqrlen( // returns maximum of all N 3-vectors in[n].
	const T *in_x, const T *in_y, const T *in_z, 
	T *out, 
	int N) 
{
	__shared__ T sh[256]; // blockDim = (256,0,0)
	T tmp1, tmp2;

	const unsigned int bid = gridDim.x * blockIdx.y + blockIdx.x;
	const unsigned int tid = threadIdx.x;
	const unsigned int i = bid * 256 + tid;

	if (i < N) {
		const T x = in_x[i];
		const T y = in_y[i];
		const T z = in_z[i];
		sh[tid] = x*x + y*y + z*z;
	} else {
		sh[tid] = 0.0f;
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
		volatile T *smem = sh;
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

template <typename T>
__global__ 
static void kernel_sum_reduce_dot_product( // sums up the dot products of all N 3-vectors in1[n] and in2[n].
	const T *in1_x, const T *in1_y, const T *in1_z,
	const T *in2_x, const T *in2_y, const T *in2_z, 
	T *out, 
	int N)
{
	__shared__ T sh[256]; // blockDim = (256,0,0)
	T tmp1, tmp2;

	const unsigned int bid = gridDim.x * blockIdx.y + blockIdx.x;
	const unsigned int tid = threadIdx.x;
	const unsigned int i = bid * 256 + tid;

	if (i < N) {
		const T x1 = in1_x[i], y1 = in1_y[i], z1 = in1_z[i];
		const T x2 = in2_x[i], y2 = in2_y[i], z2 = in2_z[i];
		sh[tid] = x1*x2 + y1*y2 + z1*z2;
	} else {
		sh[tid] = 0.0;
	}
	__syncthreads();

	if (tid < 128) {
		tmp1 = sh[tid]; tmp2 = sh[tid+128]; sh[tid] = tmp1 + tmp2;
	} 
	__syncthreads();

	if (tid < 64) {
		tmp1 = sh[tid]; tmp2 = sh[tid+64]; sh[tid] = tmp1 + tmp2;
	}
	__syncthreads();

	if (tid < 32) {
		volatile T *smem = sh;
		tmp1 = smem[tid]; tmp2 = smem[tid+32]; smem[tid] = tmp1 + tmp2;
		tmp1 = smem[tid]; tmp2 = smem[tid+16]; smem[tid] = tmp1 + tmp2;
		tmp1 = smem[tid]; tmp2 = smem[tid+ 8]; smem[tid] = tmp1 + tmp2;
		tmp1 = smem[tid]; tmp2 = smem[tid+ 4]; smem[tid] = tmp1 + tmp2;
		tmp1 = smem[tid]; tmp2 = smem[tid+ 2]; smem[tid] = tmp1 + tmp2;
		tmp1 = smem[tid]; tmp2 = smem[tid+ 1]; smem[tid] = tmp1 + tmp2;
	}

	// write result for this block to global mem
	if (tid == 0) {
		out[bid] = sh[0];
	}
}

////////////////////////////////////////////////////////////////////////////////

template <typename T>
static void alloc_reduce_buffers(T **buf1, T **buf2, int N)
{
	if ((N+255)/256 >= 65535) {
		throw std::runtime_error("Fixme: Support more than 65536*256 cells in cuda reduce routines.");
	}

	const int buf1_size = (N         + 255) / 256;
	const int buf2_size = (buf1_size + 255) / 256;

	T *tmp;
	cudaMalloc((void**)&tmp, (buf1_size + buf2_size) * sizeof(T));
	
	*buf1 = tmp + 0;
	*buf2 = tmp + buf1_size;
}

template <typename T>
static void free_reduce_buffers(T **buf1, T **buf2)
{
	cudaFree(*buf1);
	*buf1 = 0;
	*buf2 = 0;
}

template <typename T>
static T download_real(const T *ptr)
{
	T f; 
	if (cudaSuccess != cudaMemcpy((void*)&f, (const void*)ptr, sizeof(T), cudaMemcpyDeviceToHost))
		throw std::runtime_error("Reduce operation: Could not download final result from GPU.");
	return f;
}

template <typename T>
double cuda_min_impl(const T *src, int N)
{
	T *buf1, *buf2; 
	alloc_reduce_buffers(&buf1, &buf2, N);

	// First iteration.
	{
		const int B = (N+255)/256;
		kernel_min_reduce<<<B, 256>>>(src, buf1, N);
		N = B;
	}

	// Rest iterations.
	T *in = buf1, *out = buf2;
	while (N > 1) {
		const int B = (N+255)/256;
		kernel_min_reduce<<<B, 256>>>(in, out, N);
		N = B; std::swap(in, out);
	}

	checkCudaLastError("kernel_min_reduce() execution failed");

	const double result = static_cast<double>(download_real(in)); // result is stored at in[0]
	CUDA_THREAD_SYNCHRONIZE();
	free_reduce_buffers(&buf1, &buf2);
	return result;
}

template <typename T>
double cuda_max_impl(const T *src, int N)
{
	T *buf1, *buf2; 
	alloc_reduce_buffers(&buf1, &buf2, N);

	// First iteration.
	{
		const int B = (N+255)/256;
		kernel_max_reduce<<<B, 256>>>(src, buf1, N);
		N = B;
	}

	// Rest iterations.
	T *in = buf1, *out = buf2;
	while (N > 1) {
		const int B = (N+255)/256;
		kernel_max_reduce<<<B, 256>>>(in, out, N);
		N = B; std::swap(in, out);
	}

	checkCudaLastError("kernel_max_reduce() execution failed");

	const double result = static_cast<double>(download_real(in)); // result is stored at in[0]
	CUDA_THREAD_SYNCHRONIZE();
	free_reduce_buffers(&buf1, &buf2);
	return result;
}

template <typename T>
double cuda_sum_impl(const T *src, int N)
{
	T *buf1, *buf2; 
	alloc_reduce_buffers(&buf1, &buf2, N);

	// First iteration.
	{
		const int B = (N+255)/256;
		kernel_sum_reduce<<<B, 256>>>(src, buf1, N);
		N = B;
	}

	// Rest iterations.
	T *in = buf1, *out = buf2;
	while (N > 1) {
		const int B = (N+255)/256;
		kernel_sum_reduce<<<B, 256>>>(in, out, N);
		N = B; std::swap(in, out);
	}

	checkCudaLastError("kernel_sum_reduce() execution failed");

	const double result = static_cast<double>(download_real(in)); // result is stored at in[0]
	CUDA_THREAD_SYNCHRONIZE();
	free_reduce_buffers(&buf1, &buf2);
	return result;
}

template <typename T>
double cuda_absmax3_impl(const T *src_x, const T *src_y, const T *src_z, int N)
{
	T *buf1, *buf2; 
	alloc_reduce_buffers(&buf1, &buf2, N);

	// First iteration.
	{
		const int B = (N+255)/256;
		kernel_max_reduce_sqrlen<<<B, 256>>>(src_x, src_y, src_z, buf1, N);
		N = B;
	}

	// Rest iterations.
	T *in = buf1, *out = buf2;
	while (N > 1) {
		const int B = (N+255)/256;
		kernel_max_reduce<<<B, 256>>>(in, out, N);
		N = B; std::swap(in, out);
	}

	checkCudaLastError("kernel_sum_reduce() execution failed");

	const double result = std::sqrt(static_cast<double>(download_real(in))); // result is stored at in[0]
	CUDA_THREAD_SYNCHRONIZE();
	free_reduce_buffers(&buf1, &buf2);
	return result;
}

template <typename T>
double cuda_sumdot3_impl(const T *lhs_x, const T *lhs_y, const T *lhs_z, 
                         const T *rhs_x, const T *rhs_y, const T *rhs_z, int N)
{
	T *buf1, *buf2; 
	alloc_reduce_buffers(&buf1, &buf2, N);

	// First iteration.
	{
		const int B = (N+255)/256;
		kernel_sum_reduce_dot_product<<<B, 256>>>(lhs_x, lhs_y, lhs_z, rhs_x, rhs_y, rhs_z, buf1, N);
		N = B;
	}
	checkCudaLastError("kernel_sum_reduce_dot_product() execution failed");

	// Rest iterations.
	T *in = buf1, *out = buf2;
	while (N > 1) {
		const int B = (N+255)/256;
		kernel_sum_reduce<<<B, 256>>>(in, out, N);
		N = B; std::swap(in, out);
	}

	checkCudaLastError("kernel_sum_reduce() execution failed");

	const double result = static_cast<double>(download_real(in)); // result is stored at in[0]
	CUDA_THREAD_SYNCHRONIZE();
	free_reduce_buffers(&buf1, &buf2);
	return result;
}

////////////////////////////////////////////////////////////////////////////////

double cuda_min(const float *src, int N) { return cuda_min_impl<float>(src, N); }
double cuda_max(const float *src, int N) { return cuda_max_impl<float>(src, N); }
double cuda_sum(const float *src, int N) { return cuda_sum_impl<float>(src, N); }
double cuda_absmax3(const float *src_x, const float *src_y, const float *src_z, int N) { return cuda_absmax3_impl<float>(src_x, src_y, src_z, N); }
double cuda_sumdot3(const float *lhs_x, const float *lhs_y, const float *lhs_z, 
                    const float *rhs_x, const float *rhs_y, const float *rhs_z, int N) { return cuda_sumdot3_impl<float>(lhs_x, lhs_y, lhs_z, rhs_x, rhs_y, rhs_z, N); }

#ifdef HAVE_CUDA_64
double cuda_min(const double *src, int N) { return cuda_min_impl<double>(src, N); }
double cuda_max(const double *src, int N) { return cuda_max_impl<double>(src, N); }
double cuda_sum(const double *src, int N) { return cuda_sum_impl<double>(src, N); }
double cuda_absmax3(const double *src_x, const double *src_y, const double *src_z, int N) { return cuda_absmax3_impl<double>(src_x, src_y, src_z, N); }
double cuda_sumdot3(const double *lhs_x, const double *lhs_y, const double *lhs_z, 
                    const double *rhs_x, const double *rhs_y, const double *rhs_z, int N) { return cuda_sumdot3_impl<double>(lhs_x, lhs_y, lhs_z, rhs_x, rhs_y, rhs_z, N); }
#endif
