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

#include "config.h"
#include "runge_kutta.h"
#include "runge_kutta_cpu.h"
#ifdef HAVE_CUDA
#include "runge_kutta_cuda.h"
#endif

#include "Magneto.h"

#include <stdexcept>
#include <cassert>

void rk_prepare_step(
	int step, double h, ButcherTableau &tab,
	const VectorMatrix &k0, const VectorMatrix &k1, const VectorMatrix &k2,
	const VectorMatrix &k3, const VectorMatrix &k4, const VectorMatrix &k5,
	const VectorMatrix &y,
	VectorMatrix &ytmp)
{
	if (isCudaEnabled()) {
#ifdef HAVE_CUDA
		rk_prepare_step_cuda(step, h, tab, k0, k1, k2, k3, k4, k5, y, ytmp, isCuda64Enabled());
#else
		assert(0);
#endif
	} else {
		rk_prepare_step_cpu(step, h, tab, k0, k1, k2, k3, k4, k5, y, ytmp);
	}
}

void rk_combine_result(
	double h, ButcherTableau &tab,
	const VectorMatrix &k0, const VectorMatrix &k1, const VectorMatrix &k2, const VectorMatrix &k3,
	VectorMatrix &y, VectorMatrix &y_error)
{
	const int s = y.size();
	if (   s != y_error.size()
	    || s != k1.size()
	    || s != k2.size()) throw std::runtime_error("rk_combine_result: Input matrix size mismatch.");
	if (!tab.num_steps == 3) throw std::runtime_error("Need num_steps == 3 in rk_combine_result");

	if (isCudaEnabled()) {
#ifdef HAVE_CUDA
		rk_combine_result_cuda(h, tab, k0, k1, k2, k3, y, y_error, isCuda64Enabled());
#else
		assert(0);
#endif
	} else {
		rk_combine_result_cpu(h, tab, k0, k1, k2, k3, y, y_error);
	}
}

void rk_combine_result(
	double h, ButcherTableau &tab,
	const VectorMatrix &k0, const VectorMatrix &k1, const VectorMatrix &k2,
	const VectorMatrix &k3, const VectorMatrix &k4, const VectorMatrix &k5,
	VectorMatrix &y, VectorMatrix &y_error)
{
	const int s = y.size();
	if (   s != y_error.size()
	    || s != k1.size()
	    || s != k2.size()
	    || s != k3.size()
	    || s != k4.size()
	    || s != k5.size()) throw std::runtime_error("rk_combine_result: Input matrix size mismatch.");
	if (!tab.num_steps == 6) throw std::runtime_error("Need num_steps == 6 in rk_combine_result");

	if (isCudaEnabled()) {
#ifdef HAVE_CUDA
		rk_combine_result_cuda(h, tab, k0, k1, k2, k3, k4, k5, y, y_error, isCuda64Enabled());
#else
		assert(0);
#endif
	} else {
		rk_combine_result_cpu(h, tab, k0, k1, k2, k3, k4, k5, y, y_error);
	}
}

double rk_adjust_stepsize(int order, double h, double eps_abs, double eps_rel, const VectorMatrix &y, const VectorMatrix &y_error)
{
	double norm = 0.0;

	if (isCudaEnabled()) {
#ifdef HAVE_CUDA
		assert(0 && "Not implemented for cuda: rk_adjust_stepsize!");
#endif
	} else {
		norm = rk_scaled_error_norm_cpu(h, eps_abs, eps_rel, y, y_error);
	}

	// from error norm, adjust stepsize.
	const double S = 0.9; // this is called step_headroom in OOMMF

	if (norm > 1.1) {
		// decrease step, no more than factor of 5, but a fraction S more
		// than scaling suggests (for better accuracy)
		double r = S / std::pow(norm, 1.0/order); // r = S * pow(1/norm, 1/order), 1 is the desired scaled error (norm=1 maximizes step size h within the desired error bounds)
		if (r < 0.2) r = 0.2;
		return h*r;

	} else if (norm < 0.5) {
		// increase step, but no more than by a factor of 5 
		double r = S / std::pow(norm, 1.0/(order+1.0));
		if (r > 5.0) r = 5.0; // increase no more than factor of 5
		if (r < 1.0) r = 1.0; // don't allow any decrease caused by S<1 
		return h*r;
	} else {
		// no change 
		return h;
	}
}

