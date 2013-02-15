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

#include "cpu_multiplication.h"

// Complex multiplication: res = a*b + c*d + e*f
template <typename T>
inline void mul3(
	T &res_r, T &res_i,
	T ar, T ai,
	T br, T bi,
	T cr, T ci,
	T dr, T di,
	T er, T ei,
	T fr, T fi)
{
	// a*b
	res_r = ar*br - ai*bi; res_i = ai*br + ar*bi;
	// c*d
	res_r += cr*dr - ci*di; res_i += ci*dr + cr*di;
	// e*f
	res_r += er*fr - ei*fi; res_i += ei*fr + er*fi;
}

// Complex multiplication: res = a*b + c*d
template <typename T>
inline void mul2(
	T &res_r, T &res_i,
	T ar, T ai,
	T br, T bi,
	T cr, T ci,
	T dr, T di)
{
	// a*b
	res_r = ar*br - ai*bi; res_i = ai*br + ar*bi;
	// c*d
	res_r += cr*dr - ci*di; res_i += ci*dr + cr*di;
}

void cpu_multiplication_symmetric(
	int num_elements,
	const double *Nxxr, const double *Nxyr, const double *Nxzr, const double *Nyyr, const double *Nyzr, const double *Nzzr, /*in*/
	const double *Nxxi, const double *Nxyi, const double *Nxzi, const double *Nyyi, const double *Nyzi, const double *Nzzi, /*in*/
	double *Mx, double *My, double *Mz) /*inout*/
{
	for (int n=0; n<num_elements; ++n) {
		const int m = n*2;

		const double Nxx_r = Nxxr[n], Nxx_i = Nxxi[n];
		const double Nxy_r = Nxyr[n], Nxy_i = Nxyi[n];
		const double Nxz_r = Nxzr[n], Nxz_i = Nxzi[n];
		const double Nyy_r = Nyyr[n], Nyy_i = Nyyi[n];
		const double Nyz_r = Nyzr[n], Nyz_i = Nyzi[n];
		const double Nzz_r = Nzzr[n], Nzz_i = Nzzi[n];

		const double x_r = Mx[m+0], x_i = Mx[m+1];
		const double y_r = My[m+0], y_i = My[m+1];
		const double z_r = Mz[m+0], z_i = Mz[m+1];

		mul3<double>(Mx[m+0], Mx[m+1], // Hx = 
		     x_r, x_i, Nxx_r, Nxx_i,   //      Nxx*Mx
		     y_r, y_i, Nxy_r, Nxy_i,   //    + Nxy*My
		     z_r, z_i, Nxz_r, Nxz_i);  //    + Nxz*Mz
		     
		mul3<double>(My[m+0], My[m+1], // Hy = 
		     x_r, x_i, Nxy_r, Nxy_i,   //      Nyx*Mx
		     y_r, y_i, Nyy_r, Nyy_i,   //    + Nyy*My
		     z_r, z_i, Nyz_r, Nyz_i);  //    + Nyz*Mz

		mul3<double>(Mz[m+0], Mz[m+1], // Hz = 
		     x_r, x_i, Nxz_r, Nxz_i,   //       Nzx*Mx
		     y_r, y_i, Nyz_r, Nyz_i,   //     + Nzy*My
		     z_r, z_i, Nzz_r, Nzz_i);  //     + Nzz*Mz
	}
}

void cpu_multiplication_antisymmetric(
	int num_elements,
	const double *Nxyr, const double *Nxzr, const double *Nyzr, /*in*/
	const double *Nxyi, const double *Nxzi, const double *Nyzi, /*in*/
	double *Mx, double *My, double *Mz) /*inout*/
{
	for (int n=0; n<num_elements; ++n) {
		const int m = n*2;

		const double Nxy_r = Nxyr[n], Nxy_i = Nxyi[n];
		const double Nxz_r = Nxzr[n], Nxz_i = Nxzi[n];
		const double Nyz_r = Nyzr[n], Nyz_i = Nyzi[n];

		const double x_r = Mx[m+0], x_i = Mx[m+1];
		const double y_r = My[m+0], y_i = My[m+1];
		const double z_r = Mz[m+0], z_i = Mz[m+1];

		mul2<double>(Mx[m+0], Mx[m+1],
		     y_r, y_i, +Nxy_r, +Nxy_i, 
		     z_r, z_i, +Nxz_r, +Nxz_i);
		     
		mul2<double>(My[m+0], My[m+1],
		     x_r, x_i, -Nxy_r, -Nxy_i, 
		     z_r, z_i, +Nyz_r, +Nyz_i);

		mul2<double>(Mz[m+0], Mz[m+1],
		     x_r, x_i, -Nxz_r, -Nxz_i,
		     y_r, y_i, -Nyz_r, -Nyz_i);
	}
}
