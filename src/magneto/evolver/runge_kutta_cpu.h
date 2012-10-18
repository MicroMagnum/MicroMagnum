#ifndef RUNGE_KUTTA_CPU_H
#define RUNGE_KUTTA_CPU_H

#include "config.h"
#include "runge_kutta.h"

void rk_prepare_step_cpu(
	int step,
	double h,
	ButcherTableau &tab,

	const VectorMatrix &k0,
	const VectorMatrix &k1,
	const VectorMatrix &k2,
	const VectorMatrix &k3,
	const VectorMatrix &k4,
	const VectorMatrix &k5,

	const VectorMatrix &y,
	VectorMatrix &ytmp
);

void rk_combine_result_cpu(
	double h, ButcherTableau &tab,
	const VectorMatrix &k0, const VectorMatrix &k1, const VectorMatrix &k2,
	const VectorMatrix &k3, const VectorMatrix &k4, const VectorMatrix &k5,
	VectorMatrix &y, VectorMatrix &y_error
);

void rk_combine_result_cpu(
	double h, ButcherTableau &tab,
	const VectorMatrix &k0, const VectorMatrix &k1, const VectorMatrix &k2, const VectorMatrix &k3,
	VectorMatrix &y, VectorMatrix &y_error
);

double rk_scaled_error_norm_cpu(double h, double eps_abs, double eps_rel, const VectorMatrix &y, const VectorMatrix &y_error);

#endif
