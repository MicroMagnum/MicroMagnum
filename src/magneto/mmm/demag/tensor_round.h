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

#ifndef TENSOR_ROUND_H
#define TENSOR_ROUND_H

#include "config.h"

enum {
	PADDING_DISABLE = 0,
	PADDING_ROUND_2 = 1,
	PADDING_ROUND_4 = 2,
	PADDING_ROUND_8 = 3,
	PADDING_ROUND_POT = 4,
	PADDING_SMALL_PRIME_FACTORS = 5,
};

int round2(int x);
int round4(int x);
int round8(int x);
int round_pot(int x);
int round_small_prime_factors(int x);

int round_tensor_dimension(int dim, bool periodic, int padding);

int my_mod(int x, int y);

#endif
