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

#ifndef ACCURATE_SUM_H
#define ACCURATE_SUM_H

#include "config.h"
#include <algorithm>

template <class T>
T accurate_sum(T* data, int N)
{
	std::sort(data, data+N, std::greater<T>());

	volatile T a, b, x, y, z;
	volatile T sum, err;

	sum = data[0];
	err = 0;

	for (int i=1; i<N; ++i) {
		a = sum; b = data[i];

		// [x, y] = TwoSum(a, b)
		x = a + b;
		z = x - a;
		y = (a - (x - z)) + (b - z);
		
		sum = x;
		err = err + y;
	}

	return sum + err;
}

#endif
