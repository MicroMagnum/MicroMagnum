#ifndef FDM_ZHANGLI_CUDA_H
#define FDM_ZHANGLI_CUDA_H

#include "config.h"
#include "fdm_zhangli.h"

void fdm_zhangli_cuda(
	int dim_x, int dim_y, int dim_z,
	double delta_x, double delta_y, double delta_z,
	bool do_precess,
	const Matrix &P, const Matrix &xi,
	const Matrix &Ms, const Matrix &alpha,
	const VectorMatrix &j, const VectorMatrix &M,
	VectorMatrix &dM,
	bool cuda64
);

#endif
