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
#include "demag_phi_tensor.h"
#include "demag_phi_coeff.h"
#include "mmm/constants.h"

#include <stdexcept>
#include <cstdlib>

#include "Logger.h"

#include "../tensor_round.h"

Matrix GeneratePhiDemagTensor(
	int dim_x, int dim_y, int dim_z, 
	double delta_x, double delta_y, double delta_z, 
	bool periodic_x, bool periodic_y, bool periodic_z, int periodic_repeat,
	int padding,
	const char *cache_dir)
{
	int exp_x = round_tensor_dimension(dim_x, periodic_x, padding);
	int exp_y = round_tensor_dimension(dim_y, periodic_y, padding);
	int exp_z = round_tensor_dimension(dim_z, periodic_z, padding);

	// also pad if exp_i = 1
	if (exp_x == 1) exp_x = 2;
	if (exp_y == 1) exp_y = 2;
	if (exp_z == 1) exp_z = 2;

	LOG_INFO  << "Preparing scalar potential tensor.";
	LOG_DEBUG << "  Magn. size: " << dim_x << "x" << dim_y << "x" << dim_z;
	LOG_DEBUG << "  Zeropadded: " << exp_x << "x" << exp_y << "x" << exp_z;
	LOG_DEBUG << "Periodic boundary conditions:";
	LOG_DEBUG << "  Dimensions : " << (periodic_x ? "x" : "") << (periodic_y ? "y" : "") << (periodic_z ? "z" : "");
	LOG_DEBUG << "  Repetitions: " << periodic_repeat;

	if (periodic_x || periodic_y || periodic_z) {
		throw std::runtime_error("Demag potential tensor field calculation for periodic boundary conditions not yet supported");
	}

	Matrix S(Shape(3, exp_x, exp_y, exp_z));
	S.clear();

	if (std::getenv("MAGNUM_DEMAG_GARBAGE")) {
		LOG_INFO << "Skipping phi tensor generation ('MAGNUM_GARBAGE' environment variable is set!)";
		return S;
	}

	LOG_DEBUG<< "Generating...";

	Matrix::wo_accessor S_acc(S);

	int cnt = 0, percent = 0;
	for (int z=0; z<exp_z; ++z)
	for (int y=0; y<exp_y; ++y) {
		for (int x=0; x<exp_x; ++x) {
			// position delta (in number of cells)
			const int dx = x - dim_x + 1;
			const int dy = y - dim_y + 1;
			const int dz = z - dim_z + 1;

			/*if (dx >= dim_x+1) continue;
			if (dy >= dim_y+1) continue;
			if (dz >= dim_z+1) continue;*/

			// position delta (in nanometer)
			const double diff_x = dx * delta_x - 0.5*delta_x;
			const double diff_y = dy * delta_y - 0.5*delta_y;
			const double diff_z = dz * delta_z - 0.5*delta_z;
			
			// store Nij at modulo position
			const int X = (dx + 2*exp_x) % exp_x;
			const int Y = (dy + 2*exp_y) % exp_y;
			const int Z = (dz + 2*exp_z) % exp_z;

			// insert components (all real)
			Vector3d S_i(
				- demag_phi_coeff::getS<long double>(0, diff_x, diff_y, diff_z, delta_x, delta_y, delta_z),
				- demag_phi_coeff::getS<long double>(1, diff_x, diff_y, diff_z, delta_x, delta_y, delta_z),
				- demag_phi_coeff::getS<long double>(2, diff_x, diff_y, diff_z, delta_x, delta_y, delta_z)
			);
			S_acc.at(0, X, Y, Z) = S_i.x;
			S_acc.at(1, X, Y, Z) = S_i.y;
			S_acc.at(2, X, Y, Z) = S_i.z;
		}

		cnt += 1;
		if (100 * cnt / (exp_y*exp_z) > percent) {
			LOG_DEBUG << " * " << percent << "%";
			percent += 10;
		}
	}

	LOG_INFO << "Done.";
	return S;
}
