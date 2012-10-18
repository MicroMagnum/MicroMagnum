#include "transpose.h"
#include "transpose_2d.h"
#include "transpose_3d.h"

// xyz -> Yzx
void ng_cuda_transpose_zeropad_c2c(
	cudaStream_t s0, cudaStream_t s1,
	int dim_x, int dim_y, int dim_z, // input size
	int exp_y, // exp_y >= dim_y
	const float *in, // size: dim_x * dim_y * dim_z
	      float *out // size: exp_y * dim_z * dim_x
)
{
	const bool is_2d = (dim_z == 1);
	if (is_2d) {
		ng_cuda_transpose_zeropad_c2c_2d(s0, s1, dim_x, dim_y,        exp_y, in, out);
	} else {
		ng_cuda_transpose_zeropad_c2c_3d(s0, s1, dim_x, dim_y, dim_z, exp_y, in, out);
	}
}

