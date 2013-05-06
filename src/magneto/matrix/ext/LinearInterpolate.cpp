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
#include "LinearInterpolate.h"

#include <stdexcept>

namespace matty_ext {

VectorMatrix linearInterpolate(const VectorMatrix &src, Shape dest_dim)
{
	Shape src_dim = src.getShape();

	if (src_dim.getRank() != dest_dim.getRank()) {
		throw std::runtime_error("linearInterpolate: Source and destination matrices need to have the same rank.");
	}

	if (src_dim.getRank() != 3) {
		throw std::runtime_error("linearInterpolate: Fixme: Need to have matrix of rank 3");
	}

	VectorMatrix dest(dest_dim);

	VectorMatrix::      accessor dest_acc(dest);
	VectorMatrix::const_accessor  src_acc(src);

	const bool sing_x = (src_dim.getDim(0) == 1);
	const bool sing_y = (src_dim.getDim(1) == 1);
	const bool sing_z = (src_dim.getDim(2) == 1);

	Vector3d scale(1.0, 1.0, 1.0);
	if (!sing_x) scale.x = double(dest_dim.getDim(0)-1) / double(src_dim.getDim(0)-1);
	if (!sing_y) scale.y = double(dest_dim.getDim(1)-1) / double(src_dim.getDim(1)-1);
	if (!sing_z) scale.z = double(dest_dim.getDim(2)-1) / double(src_dim.getDim(2)-1);

	for (int k=0; k<dest_dim.getDim(2); ++k)
	for (int j=0; j<dest_dim.getDim(1); ++j)
	for (int i=0; i<dest_dim.getDim(0); ++i) {
		// (x,y,z): coordinates of point with indices (i,j,k) in dst matrix
		const double x = i / scale.x;
		const double y = j / scale.y;
		const double z = k / scale.z;

		const double u = x - std::floor(x);
		const double v = y - std::floor(y);
		const double w = z - std::floor(z);

		const int I = std::floor(x);
		const int J = std::floor(y);
		const int K = std::floor(z);

		Vector3d tmp(0.0, 0.0, 0.0);
		if (true                         ) tmp = tmp + (1.0-u) * (1.0-v) * (1.0-w) * src_acc.get(I  , J  , K  );
		if (                      !sing_z) tmp = tmp + (1.0-u) * (1.0-v) *      w  * src_acc.get(I  , J  , K+1);
		if (           !sing_y           ) tmp = tmp + (1.0-u) *      v  * (1.0-w) * src_acc.get(I  , J+1, K  );
		if (           !sing_y && !sing_z) tmp = tmp + (1.0-u) *      v  *      w  * src_acc.get(I  , J+1, K+1);
		if (!sing_x                      ) tmp = tmp +      u  * (1.0-v) * (1.0-w) * src_acc.get(I+1, J  , K  );
		if (!sing_x            && !sing_z) tmp = tmp +      u  * (1.0-v) *      w  * src_acc.get(I+1, J  , K+1);
		if (!sing_x && !sing_y           ) tmp = tmp +      u  *      v  * (1.0-w) * src_acc.get(I+1, J+1, K  );
		if (!sing_x && !sing_y && !sing_z) tmp = tmp +      u  *      v  *      w  * src_acc.get(I+1, J+1, K+1);
		dest_acc.set(i, j, k, tmp);
	}

	return dest;
}

Matrix linearInterpolate(const Matrix &src, Shape dest_dim)
{
	Shape src_dim = src.getShape();

	if (src_dim.getRank() != dest_dim.getRank()) {
		throw std::runtime_error("linearInterpolate: Source and destination matrices need to have the same rank.");
	}

	if (src_dim.getRank() != 3) {
		throw std::runtime_error("linearInterpolate: Fixme: Need to have matrix of rank 3");
	}

	Matrix dest(dest_dim);

	Matrix::wo_accessor dest_acc(dest);
	Matrix::ro_accessor  src_acc(src);

	const bool sing_x = (src_dim.getDim(0) == 1);
	const bool sing_y = (src_dim.getDim(1) == 1);
	const bool sing_z = (src_dim.getDim(2) == 1);

	Vector3d scale(1.0, 1.0, 1.0);
	if (!sing_x) scale.x = double(dest_dim.getDim(0)-1) / double(src_dim.getDim(0)-1);
	if (!sing_y) scale.y = double(dest_dim.getDim(1)-1) / double(src_dim.getDim(1)-1);
	if (!sing_z) scale.z = double(dest_dim.getDim(2)-1) / double(src_dim.getDim(2)-1);

	for (int k=0; k<dest_dim.getDim(2); ++k)
	for (int j=0; j<dest_dim.getDim(1); ++j)
	for (int i=0; i<dest_dim.getDim(0); ++i) {
		// (x,y,z): coordinates of point with indices (i,j,k) in dst matrix
		const double x = i / scale.x;
		const double y = j / scale.y;
		const double z = k / scale.z;

		const double u = x - std::floor(x);
		const double v = y - std::floor(y);
		const double w = z - std::floor(z);

		const int I = std::floor(x);
		const int J = std::floor(y);
		const int K = std::floor(z);

		double tmp(0.0);
		if (true                         ) tmp = tmp + (1.0-u) * (1.0-v) * (1.0-w) * src_acc.at(I  , J  , K  );
		if (                      !sing_z) tmp = tmp + (1.0-u) * (1.0-v) *      w  * src_acc.at(I  , J  , K+1);
		if (           !sing_y           ) tmp = tmp + (1.0-u) *      v  * (1.0-w) * src_acc.at(I  , J+1, K  );
		if (           !sing_y && !sing_z) tmp = tmp + (1.0-u) *      v  *      w  * src_acc.at(I  , J+1, K+1);
		if (!sing_x                      ) tmp = tmp +      u  * (1.0-v) * (1.0-w) * src_acc.at(I+1, J  , K  );
		if (!sing_x            && !sing_z) tmp = tmp +      u  * (1.0-v) *      w  * src_acc.at(I+1, J  , K+1);
		if (!sing_x && !sing_y           ) tmp = tmp +      u  *      v  * (1.0-w) * src_acc.at(I+1, J+1, K  );
		if (!sing_x && !sing_y && !sing_z) tmp = tmp +      u  *      v  *      w  * src_acc.at(I+1, J+1, K+1);
		dest_acc.at(i, j, k) = tmp;
	}

	return dest;
}

} // ns
