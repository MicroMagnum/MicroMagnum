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

#ifndef CUDA_ARRAY_H
#define CUDA_ARRAY_H

#include "config.h"

#include "../Array.h"

namespace matty {

class CU32Array : public Array
{
	friend class CU32Device;
public:
	CU32Array(const Shape &shape, class CU32Device *device);
	virtual ~CU32Array();

	float *ptr() { return data; }
	const float *ptr() const { return data; }
	int getNumBytes() const { return getShape().getNumEl() * sizeof(float); }

private:
	float *data;
	class CU32Device *cuda_device;
};

#ifdef HAVE_CUDA_64
class CU64Array : public Array
{
	friend class CU64Device;
public:
	CU64Array(const Shape &shape, class CU64Device *device);
	virtual ~CU64Array();

	double *ptr() { return data; }
	const double *ptr() const { return data; }
	int getNumBytes() const { return getShape().getNumEl() * sizeof(double); }

private:
	double *data;
	class CU64Device *cuda_device;
};
#endif

} // ns

#endif
