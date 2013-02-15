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
#include "kernels_simple.h"

#include "matrix/device/cuda_tools.h"

///////////////////////////////////////////////////////////////////////////////////////////////
// COMPONENT-WISE MATRIX OPERATIONS                                                          //
///////////////////////////////////////////////////////////////////////////////////////////////

// Strategie:
//   gridSize  = Anzahl Multiprozessoren oder besser vielfaches davon
//   blockSize = Vielfaches von Anzahl Prozessoren pro Multiprozessor 
//                 * laut CUDA-Guide ist 256 beliebt
//                 * beste Karte hat 32 
//                 * maximum erlaubt ist 512

static const int GRID_SIZE = 32;
static const int BLOCK_SIZE = 128;

template <typename real>
__global__ static void kernel_fill(real *dst, real value, int N)
{
	const int tid     = blockDim.x * blockIdx.x + threadIdx.x;
	const int threadN = blockDim.x * gridDim.x;

	for (int i=tid; i<N; i+=threadN) {
		dst[i] = value;
	}
}

template <typename real>
__global__ static void kernel_mul(real *dst, const real *src, int N)
{
	const int     tid = blockDim.x * blockIdx.x + threadIdx.x;
	const int threadN = blockDim.x * gridDim.x;

	for (int i=tid; i<N; i+=threadN) {
		dst[i] *= src[i];
	}
}

template <typename real>
__global__ static void kernel_div(real *dst, const real *src, int N)
{
	const int     tid = blockDim.x * blockIdx.x + threadIdx.x;
	const int threadN = blockDim.x * gridDim.x;

	for (int i=tid; i<N; i+=threadN) {
		dst[i] /= src[i];
	}
}

template <typename real>
__global__ static void kernel_normalize3(real *dst_x, real *dst_y, real *dst_z, real len, int N)
{
	const int     tid = blockDim.x * blockIdx.x + threadIdx.x;
	const int threadN = blockDim.x * gridDim.x;

	for (int i=tid; i<N; i+=threadN) {
		const real x = dst_x[i], y = dst_y[i], z = dst_z[i];
		const real norm = sqrt(x*x+y*y+z*z);
		real scale = len / norm; if (norm == 0) scale = 0;
		dst_x[i] = x * scale;
		dst_y[i] = y * scale;
		dst_z[i] = z * scale;
	}
}

template <typename real>
__global__ static void kernel_normalize3(real *dst_x, real *dst_y, real *dst_z, const real *len, int N)
{
	const int     tid = blockDim.x * blockIdx.x + threadIdx.x;
	const int threadN = blockDim.x * gridDim.x;

	for (int i=tid; i<N; i+=threadN) {
		const real x = dst_x[i], y = dst_y[i], z = dst_z[i];
		const real norm = sqrt(x*x+y*y+z*z);
		real scale = len[i] / norm; if (norm == 0) scale = 0;
		dst_x[i] = x * scale;
		dst_y[i] = y * scale;
		dst_z[i] = z * scale;
	}
}

////////////////////////////////////////////////////////////////////////////////

template <typename real>
void cuda_fill_impl(real *dst, real value, int N)
{
	kernel_fill<real><<<GRID_SIZE, BLOCK_SIZE>>>(dst, value, N);
	checkCudaLastError("kernel_fill() execution failed");
	CUDA_THREAD_SYNCHRONIZE();
}

template <typename real>
void cuda_mul_impl(real *dst, const real *src, int N) // dst[i] <-- dst[i]*src[i]
{
	kernel_mul<real><<<GRID_SIZE, BLOCK_SIZE>>>(dst, src, N);
	checkCudaLastError("kernel_real_mul() execution failed");
	CUDA_THREAD_SYNCHRONIZE();
}

template <typename real>
void cuda_div_impl(real *dst, const real *src, int N) // dst[i] <-- dst[i]/src[i]
{
	kernel_div<real><<<GRID_SIZE, BLOCK_SIZE>>>(dst, src, N);
	checkCudaLastError("kernel_real_div() execution failed");
	CUDA_THREAD_SYNCHRONIZE();
}

template <typename real>
void cuda_normalize3_impl(real *x0, real *x1, real *x2, real len, int N)
{
	kernel_normalize3<real><<<GRID_SIZE, BLOCK_SIZE>>>(x0, x1, x2, len, N);
	checkCudaLastError("kernel_normalize3() execution failed (1)");
	CUDA_THREAD_SYNCHRONIZE();
}

template <typename real>
void cuda_normalize3_impl(real *x0, real *x1, real *x2, const real *len, int N)
{
	kernel_normalize3<real><<<GRID_SIZE, BLOCK_SIZE>>>(x0, x1, x2, len, N);
	checkCudaLastError("kernel_normalize3() execution failed (2)");
	CUDA_THREAD_SYNCHRONIZE();
}

////////////////////////////////////////////////////////////////////////////////

void cuda_fill(float *dst, float value, int N) { cuda_fill_impl<float>(dst, value, N); }
void cuda_mul(float *dst, const float *src, int N) { cuda_mul_impl<float>(dst, src, N); }
void cuda_div(float *dst, const float *src, int N) { cuda_div_impl<float>(dst, src, N); }
void cuda_normalize3(float *x0, float *x1, float *x2, float len, int N) { cuda_normalize3_impl<float>(x0, x1, x2, len, N); }
void cuda_normalize3(float *x0, float *x1, float *x2, const float *len, int N) { cuda_normalize3_impl<float>(x0, x1, x2, len, N); }

#ifdef HAVE_CUDA_64
void cuda_fill(double *dst, double value, int N) { cuda_fill_impl<double>(dst, value, N); }
void cuda_mul(double *dst, const double *src, int N) { cuda_mul_impl<double>(dst, src, N); }
void cuda_div(double *dst, const double *src, int N) { cuda_div_impl<double>(dst, src, N); }
void cuda_normalize3(double *x0, double *x1, double *x2, double len, int N) { cuda_normalize3_impl<double>(x0, x1, x2, len, N); }
void cuda_normalize3(double *x0, double *x1, double *x2, const double *len, int N) { cuda_normalize3_impl<double>(x0, x1, x2, len, N); }
#endif
