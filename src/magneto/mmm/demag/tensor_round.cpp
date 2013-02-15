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
#include "tensor_round.h"

#include <stdexcept>
#include <vector>
#include <algorithm>

int round2(int x)
{
	if (x&1) return x+1; else return x;
}

int round4(int x)
{
	if (x&3) return ~(~x|3)+4; else return x;
}

int round8(int x)
{
	if (x&7) return ~(~x|7)+8; else return x;
}

int round_pot(int x)
{
	x--;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	x++;
	return x;
}

static int smallest_divisor(int M)
{
	for (int i=2; i<M; ++i) {
		if (M % i == 0) return i;
	}
	return M;
}

static std::vector<int> factorize(const int N)
{
	std::vector<int> factors;

	int n = N;
	while (n > 1) {
		const int n0 = smallest_divisor(n);
		factors.push_back(n0);
		n /= n0;
	}

	return factors;
}

static int maximum_prime_factor(const int N)
{
	if (N <= 1) return 1;
	std::vector<int> factors = factorize(N);
	return *std::max_element(factors.begin(), factors.end());
}

int round_small_prime_factors(int x)
{
	const int smallest_allowed = 5;
	while (maximum_prime_factor(x) > smallest_allowed) ++x;
	return x;
}

int round_tensor_dimension(int dim, bool periodic, int padding) 
{
	if (periodic) return dim;

	const int tmp = 2*dim-1;
	if (tmp == 1) return 1;

	switch (padding) {
		case PADDING_DISABLE:             return tmp;
		case PADDING_ROUND_2:             return round2(tmp);
		case PADDING_ROUND_4:             return round4(tmp);
		case PADDING_ROUND_8:             return round8(tmp);
		case PADDING_ROUND_POT:           return round_pot(tmp);
		case PADDING_SMALL_PRIME_FACTORS: return round_small_prime_factors(tmp+1);
		default: throw std::runtime_error("Invalid padding strategy");
	}
}

int my_mod(int x, int y)
{
	const int tmp = x % y;
	if (tmp >= 0) {
		return tmp;
	} else {
		return tmp+y;
	}
}
