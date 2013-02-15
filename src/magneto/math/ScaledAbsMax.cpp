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
#include "ScaledAbsMax.h"

#ifdef HAVE_CUDA
#include "ScaledAbsMax_cuda.h"
#endif

#include <cassert>
#include "Magneto.h"

#include <iostream>
using namespace std;

static double scaled_abs_max_cpu(VectorMatrix &M, Matrix &scale)
{
	VectorMatrix::const_accessor M_acc(M);
	Matrix::ro_accessor scale_acc(scale);

	assert(M.size() == scale.size());

	double squared_abs_max = 0.0;
	for (int i=0; i<M.size(); ++i) {
		const Vector3d &m = M_acc.get(i);
		const double s = scale_acc.at(i); if (s == 0.0) continue;
		const double squared_abs = (m.x*m.x + m.y*m.y + m.z*m.z) / (s*s);
		if (squared_abs > squared_abs_max) {
			squared_abs_max = squared_abs;
		}
	}
	return std::sqrt(squared_abs_max);
}

double scaled_abs_max(VectorMatrix &M, Matrix &scale)
{
	if (scale.isUniform()) {
		return M.absMax() / scale.getUniformValue();
	} else {
		const bool use_cuda = isCudaEnabled();
		if (use_cuda) {
#ifdef HAVE_CUDA
			return scaled_abs_max_cuda(M, scale, isCuda64Enabled());
#else
			assert(0);
#endif
		} else {
			return scaled_abs_max_cpu(M, scale);
		}
	}
}
