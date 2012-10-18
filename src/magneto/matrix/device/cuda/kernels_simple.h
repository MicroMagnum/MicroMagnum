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
