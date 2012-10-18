#ifndef GRADIENT_CUDA_H
#define GRADIENT_CUDA_H

#include "config.h"
#include "matrix/matty.h"

void gradient_cuda(double delta_x, double delta_y, double delta_z, const Matrix &pot, VectorMatrix &field);

#endif
