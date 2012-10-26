/*
 * Copyright 2012 by the Micromagnum authors.
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
#include "fdm_exchange.h"
#include "fdm_exchange_cpu.h"
#ifdef HAVE_CUDA
#include "fdm_exchange_cuda.h"
#include <cuda_runtime.h>
#endif

#include "Magneto.h"
#include "Benchmark.h"

double fdm_exchange(
	int dim_x, int dim_y, int dim_z,
	double delta_x, double delta_y, double delta_z,
	bool periodic_x, bool periodic_y, bool periodic_z,
	const Matrix &Ms,
	const Matrix &A,
	const VectorMatrix &M,
	VectorMatrix &H)
{
	const bool use_cuda = isCudaEnabled();

	double res = 0;
	if (use_cuda) {
#ifdef HAVE_CUDA
		CUTIC("exchange");
		res = fdm_exchange_cuda(dim_x, dim_y, dim_z, delta_x, delta_y, delta_z, periodic_x, periodic_y, periodic_z, Ms, A, M, H, isCuda64Enabled());
		CUTOC("exchange");
#else
		assert(0);
#endif
	} else {
		TIC("exchange");
		res = fdm_exchange_cpu(dim_x, dim_y, dim_z, delta_x, delta_y, delta_z, periodic_x, periodic_y, periodic_z, Ms, A, M, H);
		TOC("exchange");
	}

	return res;
}
