#ifndef CUDA_FDM_EXCHANGE_H
#define CUDA_FDM_EXCHANGE_H

#include "config.h"
#include "matrix/matty.h"

double fdm_exchange_cuda(
	int dim_x, int dim_y, int dim_z,
	double delta_x, double delta_y, double delta_z,
	bool periodic_x, bool periodic_y, bool periodic_z,
	const Matrix &Ms,
	const Matrix &A,
	const VectorMatrix &M,
	VectorMatrix &H,
	bool cuda64
);

#endif
