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

#ifndef MATTY_SHAPE_H
#define MATTY_SHAPE_H

#include "config.h"
#include <vector>

namespace matty {

class Shape
{
public:
	Shape();
	Shape(int x0);
	Shape(int x0, int x1);
	Shape(int x0, int x1, int x2);
	Shape(int x0, int x1, int x2, int x3);
	~Shape();

	int operator[](int d) const { return getDim(d); }

	// get linear index (unsafe! they assume that shape rank is high enough)
	int getLinIdx(int x0) const;
	int getLinIdx(int x0, int x1) const;
	int getLinIdx(int x0, int x1, int x2) const;
	int getLinIdx(int x0, int x1, int x2, int x3) const;

	// get dimensions
	int getDim(int d) const { return dims[d]; }
	const std::vector<int> &getDims() const { return dims; }

	// get strides
	int getStride(int d) const { return strides[d]; }
	const std::vector<int> &getStrides() const { return strides; }

	inline int getRank() const { return dims.size(); }
  inline int getNumEl() const { return num_el; }

	bool sameDims(const Shape &other) const;


private:
  int num_el;
	std::vector<int> dims;
	std::vector<int> strides;

	void init(int *beg, int *end);
};

inline int Shape::getLinIdx(int x0) const
{
	return strides[0] * x0;
}

inline int Shape::getLinIdx(int x0, int x1) const
{
	return strides[0] * x0 + strides[1] * x1;
}

inline int Shape::getLinIdx(int x0, int x1, int x2) const
{
	return strides[0] * x0 + strides[1] * x1 + strides[2] * x2;
}

inline int Shape::getLinIdx(int x0, int x1, int x2, int x3) const
{
	return strides[0] * x0 + strides[1] * x1 + strides[2] * x2 + strides[3] * x3;
}

} //ns

#endif
