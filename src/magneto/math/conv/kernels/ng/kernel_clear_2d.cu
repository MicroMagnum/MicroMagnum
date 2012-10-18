#include "kernels.h"

__global__
void kernel_clear_2d(const int dim_x, const int dim_y, float *out, int out_stride_y)
{
	const int x = threadIdx.x + 16 * blockIdx.x;
	const int y = threadIdx.y + 16 * blockIdx.y;

	const bool at_border = (blockIdx.x == gridDim.x-1 || blockIdx.y == gridDim.y-1);
	if (!at_border) {
#if 0
		const int out_stride_x = 1;
		out[2*(x*out_stride_x+y*out_stride_y)+0] = 0.0f; // clear real part
		out[2*(x*out_stride_x+y*out_stride_y)+1] = 0.0f; // clear imag part
#elif 0
		// (1)
		out[2*x+0 + 2*y*out_stride_y] = 0.0f;
		out[2*x+1 + 2*y*out_stride_y] = 0.0f;
#elif 1
		// (2) Final
		out += 2*y*out_stride_y;
		out[x+ 0] = 0.0f;
		out[x+16] = 0.0f;
#else
#		error "Need to select implementation!"
#endif
	} else {
		if (x < dim_x && y < dim_y) {
			out += 2*(x+y*out_stride_y);
			out[0] = 0.0f;
			out[1] = 0.0f;
		}
	}
}

