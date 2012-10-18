#ifndef GRADIENT_H
#define GRADIENT_H

#include "config.h"
#include "matrix/matty.h"

// pot: (dim_x+1) * (dim_y+1) * (dim_z+1)
// field: dim_x * dim_y * dim_z
void gradient(double delta_x, double delta_y, double delta_z, const Matrix &pot, VectorMatrix &field);

void gradient_cpu(double delta_x, double delta_y, double delta_z, const double *phi, VectorMatrix &field);
#ifdef HAVE_CUDA
// defined in Gradient_cuda.cu
void gradient_cuda(double delta_x, double delta_y, double delta_z, const float *phi, VectorMatrix &field);
#endif

#endif
