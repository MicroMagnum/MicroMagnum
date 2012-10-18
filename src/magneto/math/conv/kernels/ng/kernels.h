#ifndef NG_KERNELS_H
#define NG_KERNELS_H

#include <cuda.h>
#include "matrix/device/cuda_tools.h"

// kernel_clear_2d.cu
__global__
void kernel_clear_2d(
	const int dim_x, const int dim_y,
	float *out, int out_stride_y
);

// kernel_transpose_3d.h
__global__
void kernel_clear_3d(
	const int dim_x, const int dim_y, const int dim_z, 
	float *out, int out_stride_y, int out_stride_z,
	const int gridDim_y, const int gridDim_z
);

// kernel_transpose_2d.cu
__global__
void kernel_transpose_2d( // XY -> YX
	const int dim_x, const int dim_y,
	const float * in, const int  in_stride_y,  
	      float *out, const int out_stride_y
);

// kernel_transpose_3d.cu
__global__
void kernel_rotate_left_3d( // XYZ --> YZX
	const int dim_x, const int dim_y, const int dim_z,
	const float * in, const int  in_stride_y, const int  in_stride_z,  
	      float *out, const int out_stride_y, const int out_stride_z, 
	const int gridSize_y, const int gridSize_z
);
__global__
void kernel_rotate_right_3d( // XYZ --> ZXY
	const int dim_x, const int dim_y, const int dim_z,
	const float * in, const int  in_stride_y, const int  in_stride_z,  
	      float *out, const int out_stride_y, const int out_stride_z,
	const int gridSize_y, const int gridSize_z
);

void configure_transpose_kernels();

#endif
