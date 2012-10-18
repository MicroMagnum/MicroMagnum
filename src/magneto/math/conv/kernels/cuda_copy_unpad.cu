#include "cuda_copy_unpad.h"

#include <cuda.h>
#include "matrix/device/cuda_tools.h"

#include <cassert>

template <typename real>
__global__
void kernel_copy_unpad_r2r_2d(
	int dim_x, int dim_y,
	int red_x,
	const float * in_x, const float * in_y, const float * in_z,
	       real *out_x,        real *out_y,        real *out_z)
{
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x < red_x && y < dim_y) {
		const int  in_idx = 1*(x + y*dim_x);
		const int out_idx = 1*(x + y*red_x);
		out_x[out_idx] = in_x[in_idx];
		out_y[out_idx] = in_y[in_idx];
		out_z[out_idx] = in_z[in_idx];
	}
}

template <typename real>
__global__
void kernel_copy_unpad_r2r_3d(
	int dim_x, int dim_y, int dim_z,
	int red_x,
	const float *in, real *out)
{
	// This thread block copies the row 
	// from (0, row_y, row_z) to (dim_x, row_y, row_z) and pads up to (exp_x, row_y, row_z)
	const int row_y = blockIdx.x;
	const int row_z = blockIdx.y;

	const int tidx = threadIdx.x; // thread idx in block
	const int tnum = blockDim.x;  // numbers of threads in block

	in  += row_y*dim_x + row_z*dim_x*dim_y;
	out += row_y*red_x + row_z*red_x*dim_y;

	// copy
	for (int i=tidx; i<red_x; i+=tnum) {
		out[i] = in[i];
	}
}

template <typename real>
void cuda_copy_unpad_r2r_impl(
	int dim_x, int dim_y, int dim_z,
	int red_x,
	const float  *in_x, const float *in_y, const float  *in_z,
	       real *out_x,       real *out_y,        real *out_z)
{
	if (dim_z == 1) {
		const dim3 threads_per_block(16, 16, 1);
		const dim3 num_blocks((red_x+15) / 16, (dim_y+15) / 16, 1);

		kernel_copy_unpad_r2r_2d<real><<<num_blocks, threads_per_block>>>(
			dim_x, dim_y, 
			red_x, 
			 in_x,  in_y,  in_z, 
			out_x, out_y, out_z
		);
		checkCudaLastError("kernel_copy_unpad_r2r_2d() execution failed");
	} else {
		const dim3 threads_per_block(128, 1, 1);
		const dim3 num_blocks(dim_y, dim_z);

		const float  *in[3] = { in_x,  in_y,  in_z};
		       real *out[3] = {out_x, out_y, out_z};

		for (int c=0; c<3; ++c) {
			kernel_copy_unpad_r2r_3d<real><<<num_blocks, threads_per_block>>>(
				dim_x, dim_y, dim_z,
				red_x,
				in[c],
				out[c]
			);
			checkCudaLastError("kernel_copy_unpad_r2r_3d() execution failed");
		}
	}

	CUDA_THREAD_SYNCHRONIZE();
}

template <typename real>
void cuda_copy_unpad_r2r_impl(
	int dim_x, int dim_y, int dim_z,
	int red_x,
	const float * in_x,
	       real *out_x)
{
	if (dim_z == 1) {
		assert("Implement me!" && 0);
	} else {
		const dim3 threads_per_block(128, 1, 1);
		const dim3 num_blocks(dim_y, dim_z);

		kernel_copy_unpad_r2r_3d<real><<<num_blocks, threads_per_block>>>(
			dim_x, dim_y, dim_z,
			red_x,
			in_x, out_x
		);
		checkCudaLastError("kernel_copy_unpad_r2r_3d() execution failed");
	}

	CUDA_THREAD_SYNCHRONIZE();
}

////////////////////////////////////////////////////////////////////////////////

void cuda_copy_unpad_r2r(
	int dim_x, int dim_y, int dim_z,
	int red_x,
	const float * in_x, const float * in_y, const float * in_z,
	      float *out_x,       float *out_y,       float *out_z)
{
	cuda_copy_unpad_r2r_impl<float>(dim_x, dim_y, dim_z, red_x, in_x, in_y, in_z, out_x, out_y, out_z);
}

void cuda_copy_unpad_r2r(
	int dim_x, int dim_y, int dim_z,
	int red_x,
	const float * in_x,
	      float *out_x)
{
	cuda_copy_unpad_r2r_impl<float>(dim_x, dim_y, dim_z, red_x, in_x, out_x);
}

void cuda_copy_unpad_r2r(
	int dim_x, int dim_y, int dim_z,
	int red_x,
	const  float * in_x,  const float * in_y,  const float * in_z,
	      double *out_x,       double *out_y,       double *out_z)
{
	cuda_copy_unpad_r2r_impl<double>(dim_x, dim_y, dim_z, red_x, in_x, in_y, in_z, out_x, out_y, out_z);
}

void cuda_copy_unpad_r2r(
	int dim_x, int dim_y, int dim_z,
	int red_x,
	const  float * in_x,
	      double *out_x)
{
	cuda_copy_unpad_r2r_impl<double>(dim_x, dim_y, dim_z, red_x, in_x, out_x);
}
