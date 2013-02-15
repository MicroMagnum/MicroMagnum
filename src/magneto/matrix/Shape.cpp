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
#include "Shape.h"
#include <iostream>

#include <cstddef>


namespace matty {


Shape::Shape()
{
	int dims[1];
	init(dims, dims+0);
}

Shape::Shape(int dim_x)
{
	int dims_[] = {dim_x};
	init(dims_, dims_+1);
}

Shape::Shape(int dim_x, int dim_y)
{
	int dims_[] = {dim_x, dim_y};
	init(dims_, dims_+2);
}

Shape::Shape(int dim_x, int dim_y, int dim_z)
{
	int dims_[] = {dim_x, dim_y, dim_z};
	init(dims_, dims_+3);
}

Shape::Shape(int x0, int x1, int x2, int x3)
{
	int dims_[] = {x0, x1, x2, x3};
	init(dims_, dims_+4);
}

Shape::~Shape()
{
}

void Shape::init(int *beg, int *end)
{
	dims = std::vector<int>(beg, end);

	int s = 1;
	for (std::size_t i=0; i<dims.size(); ++i) {
		strides.push_back(s);
		s *= dims[i];
	}

  num_el = s;
}

bool Shape::sameDims(const Shape &other) const
{
	if (getRank() != other.getRank()) return false;

	for (int i=0; i<getRank(); ++i) {
		if (getDim(i) != other.getDim(i)) return false;
	}
	return true;
}

} // ns
