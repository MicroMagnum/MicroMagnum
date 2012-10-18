#ifndef MATTY_CUDA_KERNELS_REDUCE_H
#define MATTY_CUDA_KERNELS_REDUCE_H

#include "config.h"

// Kernels that use the parallel reduce pattern.

// float32 versions
double cuda_sum(const float *src, int N);
double cuda_min(const float *src, int N);
double cuda_max(const float *src, int N);
double cuda_absmax3(const float *src_x, const float *src_y, const float *src_z, int N);
double cuda_sumdot3(const float *lhs_x, const float *lhs_y, const float *lhs_z, 
                    const float *rhs_x, const float *rhs_y, const float *rhs_z, int N);

#ifdef HAVE_CUDA_64
// double64 versions
double cuda_sum(const double *src, int N);
double cuda_min(const double *src, int N);
double cuda_max(const double *src, int N);
double cuda_absmax3(const double *src_x, const double *src_y, const double *src_z, int N);
double cuda_sumdot3(const double *lhs_x, const double *lhs_y, const double *lhs_z, 
                    const double *rhs_x, const double *rhs_y, const double *rhs_z, int N);
#endif

#endif
