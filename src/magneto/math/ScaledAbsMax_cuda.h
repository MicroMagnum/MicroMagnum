#ifndef SCALED_ABS_MAX_CUDA_H
#define SCALED_ABS_MAX_CUDA_H

#include "config.h"
#include "matrix/matty.h"

/*
 * Returns: max(abs(M_i) / scale_i)
 */
double scaled_abs_max_cuda(VectorMatrix &M, Matrix &scale, bool cuda64);

#endif
