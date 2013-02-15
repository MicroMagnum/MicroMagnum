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

#ifndef PYTHON_BYTEARRAY_H
#define PYTHON_BYTEARRAY_H

#include <Python.h>

#include <cstddef>

// PythonByteArray objects are copied and converted to Python bytearray objects when
// they are returned from a function/method.
// (see typemap defined in PythonByteArray.i)
//
// Objects of PythonByteArrays are copyable, and the internal byte array
// is shared between instances.

class PythonByteArray
{
public:
	PythonByteArray();
	PythonByteArray(size_t length);
	~PythonByteArray();

	char *get();
	size_t getSize(); 

private:
	// Todo: Use C++11 smart pointers...
	struct SharedArray 
	{
		SharedArray(char *ptr) : ptr(ptr), ref_cnt(0)
		{
			ref_cnt = new int;
			ref_cnt[0] = 1;
		}

		SharedArray(const SharedArray &other) : ptr(other.ptr), ref_cnt(other.ref_cnt)
		{
			ref_cnt[0] += 1;
		}

		~SharedArray() 
		{
			ref_cnt[0] -= 1;
			if (ref_cnt[0] == 0) delete [] ptr;
		}

		SharedArray &operator=(const SharedArray &other)
		{
			ptr = other.ptr;
			ref_cnt = other.ref_cnt;
			ref_cnt[0] += 1;
			return *this;
		}

		char *ptr;
		int *ref_cnt;
	};

	size_t length;
	SharedArray arr;
};

#endif
