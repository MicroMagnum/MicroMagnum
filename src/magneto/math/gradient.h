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
