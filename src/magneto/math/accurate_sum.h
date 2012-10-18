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
