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
#include "Matrix.h"
#include "Matrix_accessor.h"

#include "device/cpu/CPUArray.h"
#include "restrict.h"

namespace matty {

static const void get_strides(const Shape &shape, int n, int *strides)
{
	const int rank = shape.getRank();
	for (int i=0; i<n; ++i) {
		if (i < rank) {
			strides[i] = shape.getStride(i);
		} else {
			strides[i] = shape.getStride(rank-1);
		}
	}
}


MatrixAccessor_read_only::MatrixAccessor_read_only(const Matrix &mat) : mat(mat)
{
	mat.readLock(0);
	data = static_cast<const CPUArray*>(mat.getArray(0))->ptr();
	get_strides(mat.getShape(), 4, strides);
}

MatrixAccessor_read_only::~MatrixAccessor_read_only()
{
	mat.readUnlock(0);
}


MatrixAccessor_write_only::MatrixAccessor_write_only(Matrix &mat) : mat(mat)
{
	mat.markUninitialized(); // <-- this line is the only difference to MatrixAccessor_read_write.
	mat.writeLock(0);
	data = static_cast<CPUArray*>(mat.getArray(0))->ptr();
	get_strides(mat.getShape(), 4, strides);
}

MatrixAccessor_write_only::~MatrixAccessor_write_only()
{
	mat.writeUnlock(0);
}


MatrixAccessor_read_write::MatrixAccessor_read_write(Matrix &mat) : mat(mat) 
{
	mat.writeLock(0);
	data = static_cast<CPUArray*>(mat.getArray(0))->ptr();
	get_strides(mat.getShape(), 4, strides);
}

MatrixAccessor_read_write::~MatrixAccessor_read_write()
{
	mat.writeUnlock(0);
}

} // ns
