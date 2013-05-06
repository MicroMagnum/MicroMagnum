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
#include "demag_static.h"

#include <stdexcept>

VectorMatrix CalculateStrayfieldForCuboid(
	int dim_x, int dim_y, int dim_z,
	double delta_x, double delta_y, double delta_z,
	int mag_dir,
	Vector3d pos,
	Vector3d size,
	int infinity)
{
	// Check arguments.
	if ((infinity & INFINITE_POS_X || infinity & INFINITE_NEG_X) && size.x != 0.0) throw std::runtime_error("CalculateStrayfieldForCuboid: cuboid size in x-direction must be zero for infinite extents in x direction");
	if ((infinity & INFINITE_POS_Y || infinity & INFINITE_NEG_Y) && size.y != 0.0) throw std::runtime_error("CalculateStrayfieldForCuboid: cuboid size in y-direction must be zero for infinite extents in y direction");
	if ((infinity & INFINITE_POS_Z || infinity & INFINITE_NEG_Z) && size.z != 0.0) throw std::runtime_error("CalculateStrayfieldForCuboid: cuboid size in z-direction must be zero for infinite extents in z direction");
	if (size.x < 0.0 || size.y < 0.0 || size.z < 0.0) throw std::runtime_error("CalculateStrayfieldForCuboid: cuboid size must be positive");
	if (dim_x < 1 || dim_y < 1 || dim_z < 1) throw std::runtime_error("CalculateStrayfieldForCuboid: dim_x,y,z must be positive");
	if (!(delta_x > 0.0 && delta_y > 0.0 && delta_z > 0.0)) throw std::runtime_error("CalculateStrayfieldForCuboid: delta_x,y,z must be positive");
	if (mag_dir < 0 || mag_dir > 2) throw std::runtime_error("CalculateStrayfieldForCuboid: cuboid_mag_dir must be 0, 1, or 2");

	Vector3d p0 = pos;
	Vector3d p1 = pos + size;

	VectorMatrix H(Shape(dim_x, dim_y, dim_z));
	H.clear();

	VectorMatrix::accessor H_acc(H);
	H_acc.set(0,0,0, Vector3d(1,2,3));

	assert(0);
}

