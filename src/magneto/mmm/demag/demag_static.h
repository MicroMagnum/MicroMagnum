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

#ifndef DEMAG_STATIC_H
#define DEMAG_STATIC_H

#include "config.h"
#include "matrix/matty.h"

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
	// mesh size
	int dim_x, int dim_y, int dim_z,
	double delta_x, double delta_y, double delta_z,
	// select magnetization: 1 A/m in x, y, or z direction (0, 1, or 2).
	int cuboid_mag_dir,
	// cuboid position, size and optionally infinite extent
	Vector3d cuboid_pos, // position of cuboid edge at local (0,0,0)
	Vector3d cuboid_size, // cuboid size in x,y,z dirs
	// A combination of INFINITE_xyz flags.
	// For infinite dimensions, the corresponding cuboid_size component must be zero.
	// Not all combinations of infinite dimensions are allowed.
	int cuboid_infinity 
);

#endif
