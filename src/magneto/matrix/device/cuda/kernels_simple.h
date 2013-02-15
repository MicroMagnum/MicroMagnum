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

#ifndef MATTY_CUDA_KERNELS_SIMPLE_H
#define MATTY_CUDA_KERNELS_SIMPLE_H

#include "config.h"

void cuda_fill(float *dst, float value, int N);
void cuda_mul(float *dst, const float *src, int N);
void cuda_div(float *dst, const float *src, int N);

void cuda_normalize3(float *x0, float *x1, float *x2, float len, int N);
void cuda_normalize3(float *x0, float *x1, float *x2, const float *len, int N);

#ifdef HAVE_CUDA_64
void cuda_fill(double *dst, double value, int N);
void cuda_mul(double *dst, const double *src, int N);
void cuda_div(double *dst, const double *src, int N);

void cuda_normalize3(double *x0, double *x1, double *x2, double len, int N);
void cuda_normalize3(double *x0, double *x1, double *x2, const double *len, int N);
#endif

#endif
