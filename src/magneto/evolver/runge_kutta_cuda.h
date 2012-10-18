#ifndef RUNGE_KUTTA_CUDA_H
#define RUNGE_KUTTA_CUDA_H

#include "config.h"
#include "runge_kutta.h"
#include "matrix/matty.h"

void rk_prepare_step_cuda(
	int step, double h, ButcherTableau &tab,
	const VectorMatrix &k0, const VectorMatrix &k1, const VectorMatrix &k2,
	const VectorMatrix &k3, const VectorMatrix &k4, const VectorMatrix &k5,
	const VectorMatrix &y, VectorMatrix &ytmp, bool cuda64
);

void rk_combine_result_cuda(
	double h, ButcherTableau &tab,
	const VectorMatrix &k0, const VectorMatrix &k1, const VectorMatrix &k2,
	const VectorMatrix &k3, const VectorMatrix &k4, const VectorMatrix &k5,
	VectorMatrix &y, VectorMatrix &y_error, bool cuda64
);

void rk_combine_result_cuda(
	double h, ButcherTableau &tab,
	const VectorMatrix &k0, const VectorMatrix &k1, const VectorMatrix &k2, const VectorMatrix &k3,
	VectorMatrix &y, VectorMatrix &y_error, bool cuda64
);

#endif
