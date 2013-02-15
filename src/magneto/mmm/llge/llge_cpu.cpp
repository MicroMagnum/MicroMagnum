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
#include "llge.h"

#include <cstddef>

void llge_cpu(
	const Matrix &f1, 
	const Matrix &f2,
	const VectorMatrix &M,
	const VectorMatrix &H,
	VectorMatrix &dM)
{
	VectorMatrix::accessor dM_acc(dM);
	VectorMatrix::const_accessor M_acc(M), H_acc(H);
	Matrix::ro_accessor f1_acc(f1), f2_acc(f2);

	// Calculate LLG: dM = -gamma'*(MxH) - (alpha*gamma'/Ms)*Mx(MxH)
	const size_t N = f1.size();
	for (size_t i=0; i<N; ++i) {
		const Vector3d     M = M_acc.get(i);
		const Vector3d     H = H_acc.get(i);
		const Vector3d   MxH = cross(M, H);
		const Vector3d MxMxH = cross(M, MxH);
		dM_acc.set(i, f1_acc.at(i)*MxH + f2_acc.at(i)*MxMxH);
	}
}
