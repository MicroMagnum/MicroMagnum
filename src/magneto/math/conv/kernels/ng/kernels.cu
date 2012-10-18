#include "kernels.h"

#include <cuda_runtime_api.h>

void configure_transpose_kernels()
{
	static bool init = false;
	if (!init) {
		cudaFuncSetCacheConfig(kernel_rotate_left_3d , cudaFuncCachePreferL1);
		cudaFuncSetCacheConfig(kernel_rotate_right_3d, cudaFuncCachePreferL1);
		cudaFuncSetCacheConfig(kernel_transpose_2d, cudaFuncCachePreferShared);
		init = true;
	}
}
