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

#include <cstddef>

void minimize_cpu(
	const Matrix &f, const double h,
	const VectorMatrix &M,
	const VectorMatrix &H,
	VectorMatrix &M2)
{
	VectorMatrix::accessor M2_acc(M2);
	VectorMatrix::const_accessor M_acc(M), H_acc(H);
	Matrix::ro_accessor f_acc(f);

  // Calculate M2
	const size_t N = f.size();
	for (size_t i=0; i<N; ++i) {
		const Vector3d M   = M_acc.get(i);
		const Vector3d MxH = cross(M, H_acc.get(i));

    const double tau = h * f_acc.at(i);
    const double N   = 4 + tau*tau * MxH.abs()*MxH.abs();

    const Vector3d result(
      4*M.x + 4*tau * (MxH.y*M.z - MxH.z*M.y) + tau*tau*M.x * (+ MxH.x*MxH.x - MxH.y*MxH.y - MxH.z*MxH.z) + 2*tau*tau*MxH.x * (MxH.y*M.y + MxH.z*M.z),
      4*M.y + 4*tau * (MxH.z*M.x - MxH.x*M.z) + tau*tau*M.y * (- MxH.x*MxH.x + MxH.y*MxH.y - MxH.z*MxH.z) + 2*tau*tau*MxH.y * (MxH.z*M.z + MxH.x*M.x),
      4*M.z + 4*tau * (MxH.x*M.y - MxH.y*M.x) + tau*tau*M.z * (- MxH.x*MxH.x - MxH.y*MxH.y + MxH.z*MxH.z) + 2*tau*tau*MxH.z * (MxH.x*M.x + MxH.y*M.y)
    );

    M2_acc.set(i, result / N);
	}
}
