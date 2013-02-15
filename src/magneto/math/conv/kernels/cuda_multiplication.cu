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

#include "cuda_multiplication.h"
#include "cuda_multiplication.inc.cu"

#include <cassert>
#include <stdexcept>

#include <cuda.h>
#include "matrix/device/cuda_tools.h"

static int smallest_divisor(int n)
{
	for (int i=2; i<n; ++i) {
		if (n % i == 0) return i;
	}
	return n;
}

////// SYMMETRIC //////////////////////////////////////////////////////////////////

__global__
void kernel_multiplication_symmetric_naive(
	const float *Nxxr, const float *Nxyr, const float *Nxzr, const float *Nyyr, const float *Nyzr, const float *Nzzr, /*in*/
	const float *Nxxi, const float *Nxyi, const float *Nxzi, const float *Nyyi, const float *Nyzi, const float *Nzzi, /*in*/
	float *Mx, float *My, float *Mz, /*inout*/
	int num_elements)
{
	const int i_base = 256 * (blockIdx.x + blockIdx.y*gridDim.x);
	const int i_offs = threadIdx.x; // 0..255
	const int i      = i_base + i_offs;

	if (i < num_elements) {
		const float Nxx_re = Nxxr[i], Nxx_im = Nxxi[i];
		const float Nxy_re = Nxyr[i], Nxy_im = Nxyi[i];
		const float Nxz_re = Nxzr[i], Nxz_im = Nxzi[i];
		const float Nyy_re = Nyyr[i], Nyy_im = Nyyi[i];
		const float Nyz_re = Nyzr[i], Nyz_im = Nyzi[i];
		const float Nzz_re = Nzzr[i], Nzz_im = Nzzi[i];

		const float Mx_re = Mx[2*i+0], Mx_im = Mx[2*i+1];
		const float My_re = My[2*i+0], My_im = My[2*i+1];
		const float Mz_re = Mz[2*i+0], Mz_im = Mz[2*i+1];

		float Hx_re, Hx_im, Hy_re, Hy_im, Hz_re, Hz_im;
		symmetric_tensor_multiplication(
			Nxx_re, Nxx_im, Nxy_re, Nxy_im, Nxz_re, Nxz_im,
			Nyy_re, Nyy_im, Nyz_re, Nyz_im, Nzz_re, Nzz_im,
			 Mx_re,  Mx_im,  My_re,  My_im,  Mz_re,  Mz_im,
			&Hx_re, &Hx_im, &Hy_re, &Hy_im, &Hz_re, &Hz_im
		);

		Mx[2*i+0] = Hx_re; 
		Mx[2*i+1] = Hx_im;
		My[2*i+0] = Hy_re; 
		My[2*i+1] = Hy_im;
		Mz[2*i+0] = Hz_re; 
		Mz[2*i+1] = Hz_im;
	}
}

__global__
void kernel_multiplication_symmetric(
	const float *Nxxr, const float *Nxyr, const float *Nxzr, const float *Nyyr, const float *Nyzr, const float *Nzzr, /*in*/
	const float *Nxxi, const float *Nxyi, const float *Nxzi, const float *Nyyi, const float *Nyzi, const float *Nzzi, /*in*/
	float *Mx, float *My, float *Mz) /*inout*/
{
	extern __shared__ float shared[];

	const int i_base = 256 * (blockIdx.x + blockIdx.y*gridDim.x);
	const int i_offs = threadIdx.x; // 0..255
	const int i      = i_base + i_offs;
	const int j_base = 2*i_base;

	// Coaligned global memory access
	const float Nxx_re = Nxxr[i], Nxx_im = Nxxi[i];
	const float Nxy_re = Nxyr[i], Nxy_im = Nxyi[i];
	const float Nxz_re = Nxzr[i], Nxz_im = Nxzi[i];
	const float Nyy_re = Nyyr[i], Nyy_im = Nyyi[i];
	const float Nyz_re = Nyzr[i], Nyz_im = Nyzi[i];
	const float Nzz_re = Nzzr[i], Nzz_im = Nzzi[i];

	// Copy Mx,My,Mz to shared memory (coaligned access, no shared-mem bank conflicts)
	shared[0*256+i_offs] = Mx[0*256+j_base+i_offs];
	shared[1*256+i_offs] = Mx[1*256+j_base+i_offs];
	shared[2*256+i_offs] = My[0*256+j_base+i_offs];
	shared[3*256+i_offs] = My[1*256+j_base+i_offs];
	shared[4*256+i_offs] = Mz[0*256+j_base+i_offs];
	shared[5*256+i_offs] = Mz[1*256+j_base+i_offs];

	__syncthreads();

	// Note: shared mem bank conflicts due to i_offs*2
	// With 16 banks on G80 (e.g. Tesla C270):          thr=0 thr=1 thr=2 ... thr=7 thr=8 ...
	const float Mx_re = shared[0*256+i_offs*2+0]; // Bank   0     2     4         14    0
	const float Mx_im = shared[0*256+i_offs*2+1]; // Bank   1     3     5         15    1
	const float My_re = shared[2*256+i_offs*2+0]; // Bank   0     2     4         14    0
	const float My_im = shared[2*256+i_offs*2+1]; // Bank   1     3     5         15    1
	const float Mz_re = shared[4*256+i_offs*2+0]; // Bank   0     2     4         14    0
	const float Mz_im = shared[4*256+i_offs*2+1]; // Bank   1     3     5         15    1

	float Hx_re, Hx_im, Hy_re, Hy_im, Hz_re, Hz_im;
	symmetric_tensor_multiplication(
		Nxx_re, Nxx_im, Nxy_re, Nxy_im, Nxz_re, Nxz_im,
		Nyy_re, Nyy_im, Nyz_re, Nyz_im, Nzz_re, Nzz_im,
		 Mx_re,  Mx_im,  My_re,  My_im,  Mz_re,  Mz_im,
		&Hx_re, &Hx_im, &Hy_re, &Hy_im, &Hz_re, &Hz_im
	);

	__syncthreads();

	// Note: Again shared mem bank conflicts due to i_offs*2
	shared[0*256+i_offs*2+0] = Hx_re;
	shared[0*256+i_offs*2+1] = Hx_im;
	shared[2*256+i_offs*2+0] = Hy_re;
	shared[2*256+i_offs*2+1] = Hy_im;
	shared[4*256+i_offs*2+0] = Hz_re;
	shared[4*256+i_offs*2+1] = Hz_im;

	__syncthreads();

	// Copy shared memory to Mx,My,Mz (coaligned access, no shared-mem bank conflicts)
	Mx[0*256+j_base+i_offs] = shared[0*256+i_offs];
	Mx[1*256+j_base+i_offs] = shared[1*256+i_offs];
	My[0*256+j_base+i_offs] = shared[2*256+i_offs];
	My[1*256+j_base+i_offs] = shared[3*256+i_offs];
	Mz[0*256+j_base+i_offs] = shared[4*256+i_offs];
	Mz[1*256+j_base+i_offs] = shared[5*256+i_offs];
}

void cuda_multiplication_symmetric(
	int num_elements,
	const float *Nxxr, const float *Nxyr, const float *Nxzr, const float *Nyyr, const float *Nyzr, const float *Nzzr, /*in*/
	const float *Nxxi, const float *Nxyi, const float *Nxzi, const float *Nyyi, const float *Nyzi, const float *Nzzi, /*in*/
	float *Mx, float *My, float *Mz) /*inout*/
{
	const int smart_num_elements = 256 * (num_elements / 256);
	const int naive_num_elements = num_elements - smart_num_elements;

	assert(smart_num_elements % 256 == 0);
	assert(naive_num_elements >=   0);
	assert(naive_num_elements <  256);

	// I. Process num_elements that are dividable by 256
	if (smart_num_elements > 0) {	
		static const int MAX_GRID_LENGTH = 65535;
		
		int num_blocks_x = smart_num_elements / 256;
		int num_blocks_y = 1;
		while (num_blocks_x > MAX_GRID_LENGTH) {
			const int k = smallest_divisor(num_blocks_x);
			num_blocks_x /= k;
			num_blocks_y *= k;
			if (num_blocks_y > MAX_GRID_LENGTH) {
				throw std::runtime_error("FIXME: Dont know how to layout grid block for GPU inner multiplication.");
			}
		}

		const dim3 grid_dim(num_blocks_x, num_blocks_y, 1);
		const dim3 block_dim(256,1,1);
		const int shared_mem_size = 2*sizeof(float)*3 * 256;

		kernel_multiplication_symmetric<<<grid_dim, block_dim, shared_mem_size>>>(
			Nxxr, Nxyr, Nxzr, Nyyr, Nyzr, Nzzr,
			Nxxi, Nxyi, Nxzi, Nyyi, Nyzi, Nzzi,
			Mx, My, Mz
		);
		checkCudaLastError("kernel_multiplication() execution failed");
	}

	// II. Process the rest of the elements (255 max)
	if (naive_num_elements > 0) {
		const int grid_dim((naive_num_elements+256-1) / 256);
		const int block_dim(256);

		const int offs = smart_num_elements;
		kernel_multiplication_symmetric_naive<<<grid_dim, block_dim>>>(
			Nxxr + offs, Nxyr + offs, Nxzr + offs, Nyyr + offs, Nyzr + offs, Nzzr + offs,
			Nxxi + offs, Nxyi + offs, Nxzi + offs, Nyyi + offs, Nyzi + offs, Nzzi + offs,
			Mx + 2*offs, My + 2*offs, Mz + 2*offs,
			naive_num_elements
		);
		checkCudaLastError("kernel_multiplication_naive() execution failed");
	}

	// done.
	CUDA_THREAD_SYNCHRONIZE();
}

////// ASYMMETRIC //////////////////////////////////////////////////////////////////

__global__
void kernel_multiplication_antisymmetric_naive(
	const float *Nxyr, const float *Nxzr, const float *Nyzr, /*in*/
	const float *Nxyi, const float *Nxzi, const float *Nyzi, /*in*/
	float *Mx, float *My, float *Mz, /*inout*/
	int num_elements)
{
	const int i_base = 256 * (blockIdx.x + blockIdx.y*gridDim.x);
	const int i_offs = threadIdx.x; // 0..255
	const int i      = i_base + i_offs;

	if (i < num_elements) {
		const float Nxy_re = Nxyr[i];
		const float Nxy_im = Nxyi[i];
		const float Nxz_re = Nxzr[i];
		const float Nxz_im = Nxzi[i];
		const float Nyz_re = Nyzr[i];
		const float Nyz_im = Nyzi[i];

		const float Mx_re = Mx[2*i+0];
		const float Mx_im = Mx[2*i+1];
		const float My_re = My[2*i+0];
		const float My_im = My[2*i+1];
		const float Mz_re = Mz[2*i+0];
		const float Mz_im = Mz[2*i+1];

		float Hx_re, Hx_im, Hy_re, Hy_im, Hz_re, Hz_im;
		antisymmetric_tensor_multiplication(
			Nxy_re, Nxy_im, Nxz_re, Nxz_im, Nyz_re, Nyz_im,
			 Mx_re,  Mx_im,  My_re,  My_im,  Mz_re,  Mz_im,
			&Hx_re, &Hx_im, &Hy_re, &Hy_im, &Hz_re, &Hz_im
		);

		Mx[2*i+0] = Hx_re;
		Mx[2*i+1] = Hx_im;
		My[2*i+0] = Hy_re;
		My[2*i+1] = Hy_im;
		Mz[2*i+0] = Hz_re;
		Mz[2*i+1] = Hz_im;
	}
}

__global__
void kernel_multiplication_antisymmetric(
	const float *Nxyr, const float *Nxzr, const float *Nyzr, /*in*/
	const float *Nxyi, const float *Nxzi, const float *Nyzi, /*in*/
	float *Mx, float *My, float *Mz) /*inout*/
{
	extern __shared__ float shared[];

	const int i_base = 256 * (blockIdx.x + blockIdx.y*gridDim.x);
	const int i_offs = threadIdx.x; // 0..255
	const int i      = i_base + i_offs;
	const int j_base = 2*i_base;

	// Coaligned global memory access
	const float Nxy_re = Nxyr[i];
	const float Nxy_im = Nxyi[i];
	const float Nxz_re = Nxzr[i];
	const float Nxz_im = Nxzi[i];
	const float Nyz_re = Nyzr[i];
	const float Nyz_im = Nyzi[i];

	// Copy Mx,My,Mz to shared memory (coaligned access, no shared-mem bank conflicts)
	shared[0*256+i_offs] = Mx[0*256+j_base+i_offs];
	shared[1*256+i_offs] = Mx[1*256+j_base+i_offs];
	shared[2*256+i_offs] = My[0*256+j_base+i_offs];
	shared[3*256+i_offs] = My[1*256+j_base+i_offs];
	shared[4*256+i_offs] = Mz[0*256+j_base+i_offs];
	shared[5*256+i_offs] = Mz[1*256+j_base+i_offs];

	__syncthreads();

	// Note: shared mem bank conflicts due to i_offs*2
	// With 16 banks on G80 (e.g. Tesla C270):          thr=0 thr=1 thr=2 ... thr=7 thr=8 ...
	const float Mx_re = shared[0*256+i_offs*2+0]; // Bank   0     2     4         14    0
	const float Mx_im = shared[0*256+i_offs*2+1]; // Bank   1     3     5         15    1
	const float My_re = shared[2*256+i_offs*2+0]; // Bank   0     2     4         14    0
	const float My_im = shared[2*256+i_offs*2+1]; // Bank   1     3     5         15    1
	const float Mz_re = shared[4*256+i_offs*2+0]; // Bank   0     2     4         14    0
	const float Mz_im = shared[4*256+i_offs*2+1]; // Bank   1     3     5         15    1
 
	float Hx_re, Hx_im, Hy_re, Hy_im, Hz_re, Hz_im;
	antisymmetric_tensor_multiplication(
		Nxy_re, Nxy_im, Nxz_re, Nxz_im, Nyz_re, Nyz_im,
		 Mx_re,  Mx_im,  My_re,  My_im,  Mz_re,  Mz_im,
		&Hx_re, &Hx_im, &Hy_re, &Hy_im, &Hz_re, &Hz_im
	);

	__syncthreads();

	// Note: Again shared mem bank conflicts due to i_offs*2
	shared[0*256+i_offs*2+0] = Hx_re;
	shared[0*256+i_offs*2+1] = Hx_im;
	shared[2*256+i_offs*2+0] = Hy_re;
	shared[2*256+i_offs*2+1] = Hy_im;
	shared[4*256+i_offs*2+0] = Hz_re;
	shared[4*256+i_offs*2+1] = Hz_im;

	__syncthreads();

	// Copy shared memory to Mx,My,Mz (coaligned access, no shared-mem bank conflicts)
	Mx[0*256+j_base+i_offs] = shared[0*256+i_offs];
	Mx[1*256+j_base+i_offs] = shared[1*256+i_offs];
	My[0*256+j_base+i_offs] = shared[2*256+i_offs];
	My[1*256+j_base+i_offs] = shared[3*256+i_offs];
	Mz[0*256+j_base+i_offs] = shared[4*256+i_offs];
	Mz[1*256+j_base+i_offs] = shared[5*256+i_offs];
}

void cuda_multiplication_antisymmetric(
	int num_elements,
	const float *Nxyr, const float *Nxzr, const float *Nyzr, /*in*/
	const float *Nxyi, const float *Nxzi, const float *Nyzi, /*in*/
	float *Mx, float *My, float *Mz) /*inout*/
{
	const int smart_num_elements = 256 * (num_elements / 256);
	const int naive_num_elements = num_elements - smart_num_elements;

	assert(smart_num_elements % 256 == 0);
	assert(naive_num_elements >=   0);
	assert(naive_num_elements <  256);

	// I. Process num_elements that are dividable by 256
	if (smart_num_elements > 0) {	
		static const int MAX_GRID_LENGTH = 65535;
		
		int num_blocks_x = smart_num_elements / 256;
		int num_blocks_y = 1;
		while (num_blocks_x > MAX_GRID_LENGTH) {
			const int k = smallest_divisor(num_blocks_x);
			num_blocks_x /= k;
			num_blocks_y *= k;
			if (num_blocks_y > MAX_GRID_LENGTH) {
				throw std::runtime_error("FIXME: Dont know how to layout grid block for GPU inner multiplication.");
			}
		}

		const dim3 grid_dim(num_blocks_x, num_blocks_y, 1);
		const dim3 block_dim(256,1,1);
		const int shared_mem_size = 2*sizeof(float)*3 * 256;

		kernel_multiplication_antisymmetric<<<grid_dim, block_dim, shared_mem_size>>>(
			Nxyr, Nxzr, Nyzr,
			Nxyi, Nxzi, Nyzi,
			Mx, My, Mz
		);
		checkCudaLastError("kernel_multiplication_antisymmetric() execution failed");
	}

	// II. Process the rest of the elements (255 max)
	if (naive_num_elements > 0) {
		const int grid_dim((naive_num_elements+256-1) / 256);
		const int block_dim(256);

		const int offs = smart_num_elements;
		kernel_multiplication_antisymmetric_naive<<<grid_dim, block_dim>>>(
			Nxyr + offs, Nxzr + offs, Nyzr + offs,
			Nxyi + offs, Nxzi + offs, Nyzi + offs,
			Mx + 2*offs, My + 2*offs, Mz + 2*offs,
			naive_num_elements
		);
		checkCudaLastError("kernel_multiplication_antisymmetric_naive() execution failed");
	}

	// done.
	CUDA_THREAD_SYNCHRONIZE();
}

////// SCALAR PRODUCT //////////////////////////////////////////////////////////////////

__global__
void kernel_multiplication_scalar_product_naive(
	const float *Sxr, const float *Syr, const float *Szr, /*in*/
	const float *Sxi, const float *Syi, const float *Szi, /*in*/
	float *Mx /*inout*/, const float *My /*out*/, const float *Mz /*out*/,
	int num_elements)
{
	const int i_base = 256 * (blockIdx.x + blockIdx.y*gridDim.x);
	const int i_offs = threadIdx.x; // 0..255
	const int i      = i_base + i_offs;

	if (i < num_elements) {
		const float Sxr_ = Sxr[i];
		const float Sxi_ = Sxi[i];
		const float Syr_ = Syr[i];
		const float Syi_ = Syi[i];
		const float Szr_ = Szr[i];
		const float Szi_ = Szi[i];

		const float Mxr = Mx[2*i+0];
		const float Mxi = Mx[2*i+1];
		const float Myr = My[2*i+0];
		const float Myi = My[2*i+1];
		const float Mzr = Mz[2*i+0];
		const float Mzi = Mz[2*i+1];

		float Hxr, Hxi;
		gpu_mul3(&Hxr, &Hxi,                  // Hx =
			 Mxr, Mxi, Sxr_, Sxi_,        //    + Mx*Sx
			 Myr, Myi, Syr_, Syi_,        //    + My*Sy
			 Mzr, Mzi, Szr_, Szi_);       //    + Mz*Sz

		Mx[2*i+0] = Hxr;
		Mx[2*i+1] = Hxi;
	}
}

__global__
void kernel_multiplication_scalar_product(
	const float *Sxr, const float *Syr, const float *Szr, /*in*/
	const float *Sxi, const float *Syi, const float *Szi, /*in*/
	float *Mx /*inout*/, const float *My /*out*/, const float *Mz /*out*/)
{
	extern __shared__ float shared[];

	const int i_base = 256 * (blockIdx.x + blockIdx.y*gridDim.x);
	const int i_offs = threadIdx.x; // 0..255
	const int i      = i_base + i_offs;
	const int j_base = 2*i_base;

	// Coaligned global memory access
	const float Sxr_ = Sxr[i];
	const float Sxi_ = Sxi[i];
	const float Syr_ = Syr[i];
	const float Syi_ = Syi[i];
	const float Szr_ = Szr[i];
	const float Szi_ = Szi[i];

	// Copy Mx,My,Mz to shared memory (coaligned access, no shared-mem bank conflicts)
	shared[0*256+i_offs] = Mx[0*256+j_base+i_offs];
	shared[1*256+i_offs] = Mx[1*256+j_base+i_offs];
	shared[2*256+i_offs] = My[0*256+j_base+i_offs];
	shared[3*256+i_offs] = My[1*256+j_base+i_offs];
	shared[4*256+i_offs] = Mz[0*256+j_base+i_offs];
	shared[5*256+i_offs] = Mz[1*256+j_base+i_offs];

	__syncthreads();

	// Note: shared mem bank conflicts due to i_offs*2
	// With 16 banks on G80 (e.g. Tesla C270):          thr=0 thr=1 thr=2 ... thr=7 thr=8 ...
	const float Mxr = shared[      i_offs*2+0]; // Bank   0     2     4         14    0
	const float Mxi = shared[      i_offs*2+1]; // Bank   1     3     5         15    1
	const float Myr = shared[2*256+i_offs*2+0]; // Bank   0     2     4         14    0
	const float Myi = shared[2*256+i_offs*2+1]; // Bank   1     3     5         15    1
	const float Mzr = shared[4*256+i_offs*2+0]; // Bank   0     2     4         14    0
	const float Mzi = shared[4*256+i_offs*2+1]; // Bank   1     3     5         15    1
 
	float Hxr, Hxi;
	gpu_mul3(&Hxr, &Hxi,                  // Hx =
		 Mxr, Mxi, Sxr_, Sxi_,        //    + Mx*Sx
		 Myr, Myi, Syr_, Syi_,        //    + My*Sy
		 Mzr, Mzi, Szr_, Szi_);       //    + Mz*Sz

	__syncthreads();

	// Note: Again shared mem bank conflicts due to i_offs*2
	shared[i_offs*2+0] = Hxr;
	shared[i_offs*2+1] = Hxi;

	__syncthreads();

	// Copy shared memory to Mx,My,Mz (coaligned access, no shared-mem bank conflicts)
	Mx[0*256+j_base+i_offs] = shared[0*256+i_offs];
	Mx[1*256+j_base+i_offs] = shared[1*256+i_offs];
}

void cuda_multiplication_scalar_product(
	int num_elements,
	const float *Sxr, const float *Syr, const float *Szr, /*in*/
	const float *Sxi, const float *Syi, const float *Szi, /*in*/
	float *Mx /*inout*/, const float *My /*out*/, const float *Mz /*out*/)
{
	const int smart_num_elements = 256 * (num_elements / 256);
	const int naive_num_elements = num_elements - smart_num_elements;

	assert(smart_num_elements % 256 == 0);
	assert(naive_num_elements >=   0);
	assert(naive_num_elements <  256);

	// I. Process num_elements that are dividable by 256
	if (smart_num_elements > 0) {	
		static const int MAX_GRID_LENGTH = 65535;
		
		int num_blocks_x = smart_num_elements / 256;
		int num_blocks_y = 1;
		while (num_blocks_x > MAX_GRID_LENGTH) {
			const int k = smallest_divisor(num_blocks_x);
			num_blocks_x /= k;
			num_blocks_y *= k;
			if (num_blocks_y > MAX_GRID_LENGTH) {
				throw std::runtime_error("FIXME: Dont know how to layout grid block for GPU inner multiplication.");
			}
		}

		const dim3 grid_dim(num_blocks_x, num_blocks_y, 1);
		const dim3 block_dim(256,1,1);
		const int shared_mem_size = 2*sizeof(float)*3 * 256;

		kernel_multiplication_scalar_product<<<grid_dim, block_dim, shared_mem_size>>>(
			Sxr, Syr, Szr,
			Sxi, Syi, Szi,
			Mx, My, Mz
		);
		checkCudaLastError("kernel_multiplication_scalar_product() execution failed");
	}

	// II. Process the rest of the elements (255 max)
	if (naive_num_elements > 0) {
		const int grid_dim((naive_num_elements+256-1) / 256);
		const int block_dim(256);

		const int offs = smart_num_elements;
		kernel_multiplication_scalar_product_naive<<<grid_dim, block_dim>>>(
			Sxr + offs, Syr + offs, Szr + offs,
			Sxi + offs, Syi + offs, Szi + offs,
			Mx + 2*offs, My + 2*offs, Mz + 2*offs,
			naive_num_elements
		);
		checkCudaLastError("kernel_multiplication_scalar_product_naive() execution failed");
	}

	// done.
	CUDA_THREAD_SYNCHRONIZE();
}
