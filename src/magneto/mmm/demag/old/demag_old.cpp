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
#include "demag_old.h"
#include "mmm/constants.h"

#include "Logger.h"

#include "../tensor_round.h"
#include "demag_coeff.h"

namespace demag_coeff = demag_coeff_OOMMF;
//namespace demag_coeff = demag_coeff_magneto;

static void computeEntry(Matrix::wo_accessor &N_acc, int dx, int dy, int dz, double delta_x, double delta_y, double delta_z, int exp_x, int exp_y, int exp_z)
{
	// get distance (dx,dy,dz) in meters
	const double diff_x = dx * delta_x;
	const double diff_y = dy * delta_y;
	const double diff_z = dz * delta_z;

	// insert components
	const double Nxx = demag_coeff::getN<double>(0, diff_x, diff_y, diff_z, delta_x, delta_y, delta_z);
	const double Nxy = demag_coeff::getN<double>(1, diff_x, diff_y, diff_z, delta_x, delta_y, delta_z);
	const double Nxz = demag_coeff::getN<double>(2, diff_x, diff_y, diff_z, delta_x, delta_y, delta_z);
	const double Nyy = demag_coeff::getN<double>(4, diff_x, diff_y, diff_z, delta_x, delta_y, delta_z);
	const double Nyz = demag_coeff::getN<double>(5, diff_x, diff_y, diff_z, delta_x, delta_y, delta_z);
	const double Nzz = demag_coeff::getN<double>(8, diff_x, diff_y, diff_z, delta_x, delta_y, delta_z);

	// loop through all octants
	// (X,Y,Z): position in demag tensor matrix for given octants (qx,qy,qz)
	for (int qz=-1; qz<=1; qz+=2) {
		const int Z = my_mod(qz*dz, exp_z); 
		if (Z == 0 && qz != 1) continue;

		for (int qy=-1; qy<=1; qy+=2) {
			const int Y = my_mod(qy*dy, exp_y); 
			if (Y == 0 && qy != 1) continue;

			for (int qx=-1; qx<=1; qx+=2) {
				const int X = my_mod(qx*dx, exp_x); 
				if (X == 0 && qx != 1) continue;
				
				// fill octants
				N_acc.at(0, X,Y,Z) +=       Nxx; // even in x,y,z
				N_acc.at(1, X,Y,Z) += qx*qy*Nxy; // odd in x,y, even in z
				N_acc.at(2, X,Y,Z) += qx*qz*Nxz; // odd in x,z, even in y
				N_acc.at(3, X,Y,Z) +=       Nyy; // even in x,y,z
				N_acc.at(4, X,Y,Z) += qy*qz*Nyz; // odd in y,z, even in x
				N_acc.at(5, X,Y,Z) +=       Nzz; // even in x,y,z
			}
		}
	}
}

Matrix calculateDemagTensor_old(long double lx, long double ly, long double lz, int nx, int ny, int nz, int ex, int ey, int ez, int repeat_x, int repeat_y, int repeat_z)
{
	LOG_DEBUG<< "Generating.";

	Matrix N(Shape(6, ex, ey, ez)); N.fill(0.0);
	Matrix::wo_accessor N_acc(N);

	const int dx_len = repeat_x*nx-1;
	const int dy_len = repeat_y*ny-1;
	const int dz_len = repeat_z*nz-1;

	int cnt = 0, percent = 0;

	for (int dz=0; dz<=+dz_len; dz+=1) {
		for (int dy=0; dy<=+dy_len; dy+=1) {
			for (int dx=0; dx<=+dx_len; dx+=1) {
				computeEntry(N_acc, dx, dy, dz, lx, ly, lz, ex, ey, ez);
			}

			cnt += 1;
			if (100.0 * cnt / ((dy_len+1)*(dz_len+1)) > percent) {
				LOG_INFO << "  " << percent << "%";
				percent += 5;
			}
		}
	}

	LOG_INFO << "Done.";
	return N;
}
