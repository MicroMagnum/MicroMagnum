/*
 * Copyright 2012 by the Micromagnum authors.
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

%{
#include "mmm/demag/demag_tensor.h"
#include "mmm/demag/demag_static.h"
#include "mmm/demag/tensor_round.h"
#include "mmm/demag/phi/demag_phi_tensor.h"
%}

enum {
	PADDING_DISABLE = 0,
	PADDING_ROUND_2 = 1,
	PADDING_ROUND_4 = 2,
	PADDING_ROUND_8 = 3,
	PADDING_ROUND_POT = 4,
	PADDING_SMALL_PRIME_FACTORS = 5,
};

Matrix GenerateDemagTensor(
	int dim_x, int dim_y, int dim_z, 
	double delta_x, double delta_y, double delta_z, 
	bool periodic_x, bool periodic_y, bool periodic_z, int periodic_repeat,
	int padding,
	const char *cache_dir
);

Matrix GeneratePhiDemagTensor(
	int dim_x, int dim_y, int dim_z, 
	double delta_x, double delta_y, double delta_z, 
	bool periodic_x, bool periodic_y, bool periodic_z, int periodic_repeat,
	int padding,
	const char *cache_dir
);

enum {
        INFINITE_NONE=0,
        INFINITE_POS_X=(1<<0),
        INFINITE_NEG_X=(1<<1),
        INFINITE_POS_Y=(1<<2),
        INFINITE_NEG_Y=(1<<3),
        INFINITE_POS_Z=(1<<4),
        INFINITE_NEG_Z=(1<<5),
};

VectorMatrix CalculateStrayfieldForCuboid(
	int dim_x, int dim_y, int dim_z,
	double delta_x, double delta_y, double delta_z,
	int cuboid_mag_dir,
	Vector3d cuboid_pos,
	Vector3d cuboid_size,
	int cuboid_infinity 
);
