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
#include "minimize.h"
#include "minimize_cpu.h"
#ifdef HAVE_CUDA
#include "minimize_cuda.h"
#include <cuda_runtime.h>
#endif

#include "Magneto.h"
#include "Benchmark.h"

#include <cassert>

void minimize(
	const Matrix &f, const double h,
	const VectorMatrix &M,
	const VectorMatrix &H,
	VectorMatrix &M2)
{
	const bool use_cuda = isCudaEnabled();

	if (use_cuda) {
#ifdef HAVE_CUDA
		CUTIC("minimize");
#ifdef HAVE_CUDA_64
		if (isCuda64Enabled())
			minimize_cu64(f, h, M, H, M2);
		else
#endif
			minimize_cu32(f, h, M, H, M2);
		CUTOC("minimize");
#else
		assert(0);
#endif
	} else {
		TIC("minimize");
		minimize_cpu(f, h, M, H, M2);
		TOC("minimize");
	}
}
