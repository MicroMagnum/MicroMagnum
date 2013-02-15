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

#ifndef DEMAG_COEFF_H
#define DEMAG_COEFF_H

#include "config.h"
#include <math.h>
#include <cmath>
#include <cassert>
#include <algorithm>

// There are two namespaces in this file:
//   demag_coeff_OOMMF: routines taken from OOMMF source code, see demag_coeff.txt
//   demag_coeff_magneto: the same routines, but own implementation.

namespace demag_coeff_magneto
{
	template <class real>
	real newell_f(real x, real y, real z)
	{
		x = fabs(x); y = fabs(y); z = fabs(z);

		const real x2 = x*x, y2 = y*y, z2 = z*z;
		const real R = sqrt(x2 + y2 + z2);

		real result = 0;
		if (x2+z2 > 0) result += 0.5*y * (z2-x2) * std::log((y+R) / std::sqrt(x2+z2));
		if (x2+y2 > 0) result += 0.5*z * (y2-x2) * std::log((z+R) / std::sqrt(x2+y2));
		if (x * R > 0) result -= x * y * z * std::atan((y * z) / (x * R));
		result += 1.0/6.0 * (2*x2-y2-z2) * R;
		return result;
	}

	template <class real>
	real newell_g(real x, real y, real z)
	{
		z = fabs(z);

		const real x2 = x*x, y2 = y*y, z2 = z*z;
		const real R = sqrt(x2 + y2 + z2);

		real result = -(x * y * R / 3);
		if (x2+y2 > 0)       result += (x * y * z)             * std::log((z+R) / std::sqrt(x2+y2));
		if (x2+z2 > 0)       result += (x / 6) * (3 * z2 - x2) * std::log((y+R) / std::sqrt(x2+z2));
		if (y2+z2 > 0)       result += (y / 6) * (3 * z2 - y2) * std::log((x+R) / std::sqrt(y2+z2));
		if (fabs(x * R) > 0) result -= ((z * x2) / 2) * std::atan((y * z) / (x * R));
		if (fabs(y * R) > 0) result -= ((z * y2) / 2) * std::atan((x * z) / (y * R));
		if (fabs(z * R) > 0) result -= ((z2 * z) / 6) * std::atan((x * y) / (z * R));
		return result;
	}

	template <class real>
	real getN(int n, real x, real y, real z, real dx, real dy, real dz)
	{
		static const int coeff[] = {
			 8,  0,  0,  0,   -4,  1,  0,  0,   -4, -1,  0,  0,   -4,  0,  1,  0,
			-4,  0, -1,  0,   -4,  0,  0,  1,   -4,  0,  0, -1,    2,  1,  1,  0,
			 2,  1, -1,  0,    2, -1,  1,  0,    2, -1, -1,  0,    2,  1,  0,  1,
			 2,  1,  0, -1,    2, -1,  0,  1,    2, -1,  0, -1,    2,  0,  1,  1,
			 2,  0,  1, -1,    2,  0, -1,  1,    2,  0, -1, -1,   -1,  1,  1,  1,
			-1,  1,  1, -1,   -1,  1, -1,  1,   -1,  1, -1, -1,   -1, -1,  1,  1,
			-1, -1,  1, -1,   -1, -1, -1,  1,   -1, -1, -1, -1
		};
		static const int coeff_len = sizeof(coeff) / sizeof(coeff[0]);

		switch (n) {
			case 0: { // Nxx
				real result = 0;
				for (int i=0; i<coeff_len; i+=4)
					result += coeff[i] * newell_f(x + coeff[i+1] * dx, y + coeff[i+2] * dy, z + coeff[i+3] * dz);
				return result / (4 * PI * dx * dy * dz);
			}
			case 4: // Nyy
				return getN(0, y, x, z, dy, dx, dz);
			case 8: // Nzz
				return getN(0, z, y, x, dz, dy, dx);
			case 1: case 3:	{ // Nxy, Nyx
				real result = 0;
				for (int i=0; i<coeff_len; i+=4)
					result += coeff[i] * newell_g(x + coeff[i+1] * dx, y + coeff[i+2] * dy, z + coeff[i+3] * dz);
				return result / (4 * PI * dx * dy * dz);
			}
			case 2: case 6: // Nxz, Nzx
				return getN(1, x, z, y, dx, dz, dy);
			case 5: case 7:	// Nyz, Nzy
				return getN(1, y, z, x, dy, dz, dx);
			default:
				assert(0);
		}
	}

} // namespace demag_coeff_magneto

namespace demag_coeff_OOMMF 
{
	template <class real>
	real selfDemagNx(real x, real y, real z)
	{ 
		// Here Hx = -Nxx.Mx (formula (16) in Newell).

		if (x <= 0.0 || y <= 0.0 || z <= 0.0) return 0.0;
		if (x == y && y == z) return 1.0/3.0;  // Special case: cube

		const real xsq  = x*x, ysq = y*y, zsq = z*z;
		const real diag = std::sqrt(xsq + ysq + zsq);
		const real mpxy = (x-y) * (x+y);
		const real mpxz = (x-z) * (x+z);

		real Nxx = 0.0;

		Nxx +=  -4 * (2*xsq*x-ysq*y-zsq*z);
		Nxx +=   4 * (xsq+mpxy)*std::sqrt(xsq+ysq);
		Nxx +=   4 * (xsq+mpxz)*std::sqrt(xsq+zsq);
		Nxx +=  -4 * (ysq+zsq)*std::sqrt(ysq+zsq);
		Nxx +=  -4 * diag*(mpxy+mpxz);

		Nxx +=  24 * x*y*z*std::atan(y*z/(x*diag));
		Nxx +=  12 * (z+y)*xsq*std::log(x);

		Nxx +=  12 * z*ysq*std::log((std::sqrt(ysq+zsq)+z)/y);
		Nxx += -12 * z*xsq*std::log(std::sqrt(xsq+zsq)+z);
		Nxx +=  12 * z*mpxy*std::log(diag+z);
		Nxx +=  -6 * z*mpxy*std::log(xsq+ysq);

		Nxx +=  12 * y*zsq*std::log((std::sqrt(ysq+zsq)+y)/z);
		Nxx += -12 * y*xsq*std::log(std::sqrt(xsq+ysq)+y);
		Nxx +=  12 * y*mpxz*std::log(diag+y);
		Nxx +=  -6 * y*mpxz*std::log(xsq+zsq);

		return Nxx / (12*PI*x*y*z);
	}

	template <class real>
	real selfDemagNy(real xsize, real ysize, real zsize)
	{ 
		return selfDemagNx(ysize, zsize, xsize); 
	}

	template <class real>
	real selfDemagNz(real xsize, real ysize, real zsize)
	{
		return selfDemagNx(zsize, xsize, ysize);
	}

	template <class real>
	real Newell_f(real x, real y, real z)
	{ 
		// asinh(t) is written as log(t+sqrt(1+t)) because the latter
		// appears easier to handle if t=y/x (for example) as x -> 0.

		// This function is even; the fabs()'s just simplify special case handling.
		x = fabs(x); real xsq = x*x;
		y = fabs(y); real ysq = y*y;
		z = fabs(z); real zsq = z*z; 

		real R = xsq + ysq + zsq;
		if (R <= 0.0) return 0.0;
		R = sqrt(R);

		// f(x,y,z)
		real sum = 0.0;

		if (z > 0.0) { 
			sum += 2*(2*xsq-ysq-zsq)*R;

			const real temp1 = x*y*z;
			if (temp1 > 0.0)
				sum += -12*temp1*atan2(y*z,x*R);

			const real temp2 = xsq+zsq;
			if (y > 0.0 && temp2 > 0.0) {
				const real dummy = log(((y+R)*(y+R)) / temp2);
				sum +=  3*y*zsq*dummy;
				sum += -3*y*xsq*dummy;
			}

			const real temp3 = xsq+ysq;
			if (temp3 > 0.0) {
				const real dummy = log(((z+R)*(z+R)) / temp3);
				sum +=  3*z*ysq*dummy;
				sum += -3*z*xsq*dummy;
			}
		} else {
			// Simplified for z==0 (useful for 2d grids)

			if (x == y) { // for cube grids
				const real K = -2.45981439737106805379; // K = 2*sqrt(2)-6*log(1+sqrt(2))
				sum += K*xsq*x;
			} else {
				sum += 2*(2*xsq-ysq)*R;
				if (y > 0.0 && x > 0.0) sum += -6*y*xsq*log((y+R)/x);
			}
		}

		return sum / 12.0;
	}

	template <class real>
	real Newell_g(real x, real y, real z)
	{ 
		// asinh(t) is written as log(t+sqrt(1+t)) because the latter
		// appears easier to handle if t=y/x (for example) as x -> 0.

		// Handle symmetries
		real result_sign = 1.0;
		if (x < 0.0) result_sign *= -1.0;  
		if (y < 0.0) result_sign *= -1.0;
		x = fabs(x); y = fabs(y); z = fabs(z);

		const real xsq  = x*x, ysq = y*y,zsq = z*z;
		const real Rsq = xsq + ysq + zsq; 
		if (Rsq <= 0.0) return 0.0;
		const real R = std::sqrt(Rsq);

		real sum = 0.0;

		sum += -2*x*y*R;

		if (z > 0.0) {
			sum += -z*zsq*std::atan2(x*y,z*R);
			sum += -3*z*ysq*std::atan2(x*z,y*R);
			sum += -3*z*xsq*std::atan2(y*z,x*R);

			const real temp1 = xsq+ysq;
			if (temp1 > 0.0)
				sum += 6*x*y*z*std::log((z+R)/std::sqrt(temp1));

			const real temp2 = ysq+zsq;
			if (temp2 > 0.0)
				sum += y*(3*zsq-ysq)*std::log((x+R)/std::sqrt(temp2));

			const real temp3 = xsq+zsq;
			if (temp3 > 0.0)
				sum += x*(3*zsq-xsq)*std::log((y+R)/std::sqrt(temp3));

		} else {
			// Simplified for z==0 (useful for 2d grids)
			if (y > 0.0) sum += -y*ysq*std::log((x+R)/y);
			if (x > 0.0) sum += -x*xsq*std::log((y+R)/x);
		}

		return result_sign * sum / 6.0;
	}

	template <class real>
	real CalculateSDA00(real x,real y,real z, real dx, real dy, real dz)
	{ 
		// Self demag term. The base routine can handle x==y==z==0, but this should be more accurate.
		if (x == 0.0 && y == 0.0 && z == 0.0) {
			return selfDemagNx(dx,dy,dz) * (4*PI*dx*dy*dz);
		}

		real result = 0.0;

		result += -1 * Newell_f(x+dx,y+dy,z+dz);
		result += -1 * Newell_f(x+dx,y-dy,z+dz);
		result += -1 * Newell_f(x+dx,y-dy,z-dz);
		result += -1 * Newell_f(x+dx,y+dy,z-dz);
		result += -1 * Newell_f(x-dx,y+dy,z-dz);
		result += -1 * Newell_f(x-dx,y+dy,z+dz);
		result += -1 * Newell_f(x-dx,y-dy,z+dz);
		result += -1 * Newell_f(x-dx,y-dy,z-dz);

		result +=  2 * Newell_f(x,y-dy,z-dz);
		result +=  2 * Newell_f(x,y-dy,z+dz);
		result +=  2 * Newell_f(x,y+dy,z+dz);
		result +=  2 * Newell_f(x,y+dy,z-dz);
		result +=  2 * Newell_f(x+dx,y+dy,z);
		result +=  2 * Newell_f(x+dx,y,z+dz);
		result +=  2 * Newell_f(x+dx,y,z-dz);
		result +=  2 * Newell_f(x+dx,y-dy,z);
		result +=  2 * Newell_f(x-dx,y-dy,z);
		result +=  2 * Newell_f(x-dx,y,z+dz);
		result +=  2 * Newell_f(x-dx,y,z-dz);
		result +=  2 * Newell_f(x-dx,y+dy,z);

		result += -4 * Newell_f(x,y-dy,z);
		result += -4 * Newell_f(x,y+dy,z);
		result += -4 * Newell_f(x,y,z-dz);
		result += -4 * Newell_f(x,y,z+dz);
		result += -4 * Newell_f(x+dx,y,z);
		result += -4 * Newell_f(x-dx,y,z);

		result +=  8 * Newell_f(x,y,z);

		return result;
	}

	template <class real>
	real CalculateSDA01(real x, real y, real z, real l, real h, real e)
	{ 
		real sum = 0.0;

		sum += -1 * Newell_g(x-l,y-h,z-e);
		sum += -1 * Newell_g(x-l,y-h,z+e);
		sum += -1 * Newell_g(x+l,y-h,z+e);
		sum += -1 * Newell_g(x+l,y-h,z-e);
		sum += -1 * Newell_g(x+l,y+h,z-e);
		sum += -1 * Newell_g(x+l,y+h,z+e);
		sum += -1 * Newell_g(x-l,y+h,z+e);
		sum += -1 * Newell_g(x-l,y+h,z-e);

		sum +=  2 * Newell_g(x,y+h,z-e);
		sum +=  2 * Newell_g(x,y+h,z+e);
		sum +=  2 * Newell_g(x,y-h,z+e);
		sum +=  2 * Newell_g(x,y-h,z-e);
		sum +=  2 * Newell_g(x-l,y-h,z);
		sum +=  2 * Newell_g(x-l,y+h,z);
		sum +=  2 * Newell_g(x-l,y,z-e);
		sum +=  2 * Newell_g(x-l,y,z+e);
		sum +=  2 * Newell_g(x+l,y,z+e);
		sum +=  2 * Newell_g(x+l,y,z-e);
		sum +=  2 * Newell_g(x+l,y-h,z);
		sum +=  2 * Newell_g(x+l,y+h,z);

		sum += -4 * Newell_g(x-l,y,z);
		sum += -4 * Newell_g(x+l,y,z);
		sum += -4 * Newell_g(x,y,z+e);
		sum += -4 * Newell_g(x,y,z-e);
		sum += -4 * Newell_g(x,y-h,z);
		sum += -4 * Newell_g(x,y+h,z);

		sum +=  8 * Newell_g(x,y,z);

		return sum;
	}

	template <class real>
	real getN(int n, real x, real y, real z, real dx, real dy, real dz)
	{
		switch (n) {
			case 0: { // Nxx
				real result = CalculateSDA00<real>(x, y, z, dx, dy, dz);
				return result / (4 * PI * dx * dy * dz);
			}
			case 4: // Nyy
				return getN(0, y, x, z, dy, dx, dz);
			case 8: // Nzz
				return getN(0, z, y, x, dz, dy, dx);
			case 1: case 3:	{ // Nxy, Nyx
				real result = CalculateSDA01<real>(x, y, z, dx, dy, dz);
				return result / (4 * PI * dx * dy * dz);
			}
			case 2: case 6: // Nxz, Nzx
				return getN(1, x, z, y, dx, dz, dy);
			case 5: case 7:	// Nyz, Nzy
				return getN(1, y, z, x, dy, dz, dx);
			default:
				assert(0);
		}
	}

} // namespace demag_coeff_OOMMF

#endif

