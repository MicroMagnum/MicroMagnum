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

#ifndef MINIMIZE_H
#define MINIMIZE_H

#include "config.h"
#include "matrix/matty.h"

// calculate: M2 = M - h * f * (M + M2)/2 ^ H(M)
void minimize(
	const Matrix &f, const double h,
	const VectorMatrix &M,
	const VectorMatrix &H,
	VectorMatrix &M2
);

#endif
