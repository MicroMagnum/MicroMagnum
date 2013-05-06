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
#include "VectorMatrix_accessor.h"
#include "VectorMatrix.h"

#include "device/cpu/CPUArray.h"

namespace matty {

VectorMatrixAccessor::VectorMatrixAccessor(VectorMatrix &mat) : mat(mat) 
{
	mat.writeLock(0); // 0 = CPUDevice!
	data_x = static_cast<CPUArray*>(mat.getArray(0, 0))->ptr();
	data_y = static_cast<CPUArray*>(mat.getArray(0, 1))->ptr();
	data_z = static_cast<CPUArray*>(mat.getArray(0, 2))->ptr();

	// Precalculate strides
	const int rank = mat.getShape().getRank();
	strides[0] = 1;
	strides[1] = strides[0] * (rank > 0 ? mat.getShape().getDim(0) : 1);
	strides[2] = strides[1] * (rank > 1 ? mat.getShape().getDim(1) : 1);
	strides[3] = strides[2] * (rank > 2 ? mat.getShape().getDim(2) : 1);
}

VectorMatrixAccessor::~VectorMatrixAccessor()
{
	mat.writeUnlock(0);
}

ConstVectorMatrixAccessor::ConstVectorMatrixAccessor(const VectorMatrix &mat) : mat(mat) 
{
	mat.readLock(0);
	data_x = static_cast<CPUArray*>(mat.getArray(0, 0))->ptr();
	data_y = static_cast<CPUArray*>(mat.getArray(0, 1))->ptr();
	data_z = static_cast<CPUArray*>(mat.getArray(0, 2))->ptr();

	// Precalculate strides
	const int rank = mat.getShape().getRank();
	strides[0] = 1;
	strides[1] = strides[0] * (rank > 0 ? mat.getShape().getDim(0) : 1);
	strides[2] = strides[1] * (rank > 1 ? mat.getShape().getDim(1) : 1);
	strides[3] = strides[2] * (rank > 2 ? mat.getShape().getDim(2) : 1);
}

ConstVectorMatrixAccessor::~ConstVectorMatrixAccessor()
{
	mat.readUnlock(0);
}

}; //ns
