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

#include "SymmetricMatrixVectorConvolution_Simple.h"
#include <cassert>

SymmetricMatrixVectorConvolution_Simple::SymmetricMatrixVectorConvolution_Simple(const Matrix &lhs, int dim_x, int dim_y, int dim_z)
	: lhs(lhs), dim_x(dim_x), dim_y(dim_y), dim_z(dim_z)
{
	assert(lhs.getShape().getDim(0) == 6);
	exp_x = lhs.getShape().getDim(1);
	exp_y = lhs.getShape().getDim(2);
	exp_z = lhs.getShape().getDim(3);
}

SymmetricMatrixVectorConvolution_Simple::~SymmetricMatrixVectorConvolution_Simple()
{
}

void SymmetricMatrixVectorConvolution_Simple::execute(const VectorMatrix &rhs, VectorMatrix &res)
{
	Matrix::ro_accessor N_acc(lhs);

	VectorMatrix::const_accessor M_acc(rhs); 
	VectorMatrix::      accessor H_acc(res);

	// H(r) = int N(r-r')*M(r') dr'

	// Hx = Nxx*Mx + Nxy*My + Nxz*Mz
	// Hy = Nyx*Mx + Nyy*My + Nyz*Mz
	// Hz = Nxz*Mx + Nyz*My + Nzz*Mz

	for (int z=0; z<dim_z; ++z)
	for (int y=0; y<dim_y; ++y)
	for (int x=0; x<dim_x; ++x) 
	{
		Vector3d H(0.0, 0.0, 0.0);

		for (int o=0; o<dim_z; ++o)
		for (int n=0; n<dim_y; ++n)
		for (int m=0; m<dim_x; ++m) 
		{
			// (X,Y,Z): position in demag tensor field matrix
			const int X = (x-m+exp_x) % exp_x;
			const int Y = (y-n+exp_y) % exp_y;
			const int Z = (z-o+exp_z) % exp_z;

			const double Nxx = N_acc.at(0,X,Y,Z);
			const double Nxy = N_acc.at(1,X,Y,Z);
			const double Nxz = N_acc.at(2,X,Y,Z);
			const double Nyy = N_acc.at(3,X,Y,Z);
			const double Nyz = N_acc.at(4,X,Y,Z);
			const double Nzz = N_acc.at(5,X,Y,Z);

			const Vector3d &M = M_acc.get(m, n, o);

			H.x += Nxx*M.x + Nxy*M.y + Nxz*M.z;
			H.y += Nxy*M.x + Nyy*M.y + Nyz*M.z;
			H.z += Nxz*M.x + Nyz*M.y + Nzz*M.z;
		}

		H_acc.set(x,y,z,H);
	}
}
