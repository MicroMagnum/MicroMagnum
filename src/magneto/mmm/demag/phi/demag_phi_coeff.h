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

#ifndef DEMAG_PHI_COEFF_H
#define DEMAG_PHI_COEFF_H

#include "config.h"
#include <cmath>
#include <cassert>

#include "mmm/constants.h"

namespace demag_phi_coeff
{
	template <class real>
	real f(real x, real y, real z)
	{
		const real r = std::sqrt(x*x + y*y + z*z);

		real result = 0;
		result -= x;
		if (z != 0) result += z * std::atan(x / z);
		if (z != 0) result -= z * std::atan((x*y) / (z*r));
		if (y != 0) result += y * (std::log(2.0) + std::log(x+r));
		if (x != 0) result += x * (std::log(2.0) + std::log(y+r));
		return result;
	}

	template <class real>
	real g(real x, real y, real z, real dx, real dy, real dz)
	{
		const real result = + f(x - dx/2, y - dy/2, z + dz/2)
        	                    + f(x + dx/2, y + dy/2, z + dz/2)
        	                    - f(x - dx/2, y + dy/2, z + dz/2)
        	                    - f(x + dx/2, y - dy/2, z + dz/2)
		                    - f(x - dx/2, y - dy/2, z - dz/2)
        	                    - f(x + dx/2, y + dy/2, z - dz/2)
        	                    + f(x - dx/2, y + dy/2, z - dz/2)
        	                    + f(x + dx/2, y - dy/2, z - dz/2);
		return - result / (4.0 * PI);
	}

	template <class real>
	real getS(int n, real x, real y, real z, real dx, real dy, real dz)
	{
		switch (n) {
			case 0: return g(y, z, x, dy, dz, dx);
			case 1: return g(z, x, y, dz, dx, dy);
			case 2: return g(x, y, z, dx, dy, dz);
		}
		assert(0);
	}

} // namespace PotentialCoeff

#endif
