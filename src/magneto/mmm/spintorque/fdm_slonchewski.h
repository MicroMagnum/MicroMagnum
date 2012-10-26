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

#ifndef FDM_SLONCHEWSKI_H
#define FDM_SLONCHEWSKI_H

#include "config.h"
#include "matrix/matty.h"

void fdm_slonchewski(
	int dim_x, int dim_y, int dim_z,
	double delta_x, double delta_y, double delta_z,
	double a_j,
	const VectorMatrix &p, // spin polarization
	const Matrix &Ms,
	const Matrix &alpha,
	const VectorMatrix &M,
	VectorMatrix &dM
);

#endif
