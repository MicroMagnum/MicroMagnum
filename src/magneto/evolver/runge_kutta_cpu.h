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
