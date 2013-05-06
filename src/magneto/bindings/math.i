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

%{
#include "math/conv/SymmetricMatrixVectorConvolution_FFT.h"
#include "math/conv/SymmetricMatrixVectorConvolution_Simple.h"
#include "math/conv/AntisymmetricMatrixVectorConvolution_FFT.h"
#include "math/conv/VectorVectorConvolution_FFT.h"
%}

class SymmetricMatrixVectorConvolution_FFT
{
public:
	SymmetricMatrixVectorConvolution_FFT(const Matrix &lhs, int dim_x, int dim_y, int dim_z);
	virtual ~SymmetricMatrixVectorConvolution_FFT();
	virtual void execute(const VectorMatrix &rhs, VectorMatrix &res);
};

class SymmetricMatrixVectorConvolution_Simple
{
public:
	SymmetricMatrixVectorConvolution_Simple(const Matrix &lhs, int dim_x, int dim_y, int dim_z);
	virtual ~SymmetricMatrixVectorConvolution_Simple();
	virtual void execute(const VectorMatrix &rhs, VectorMatrix &res);
};

class AntisymmetricMatrixVectorConvolution_FFT
{
public:
	AntisymmetricMatrixVectorConvolution_FFT(const Matrix &lhs, int dim_x, int dim_y, int dim_z);
	virtual ~AntisymmetricMatrixVectorConvolution_FFT();
	virtual void execute(const VectorMatrix &rhs, VectorMatrix &res);
};

class VectorVectorConvolution_FFT
{
public:
	VectorVectorConvolution_FFT(const Matrix &lhs, int dim_x, int dim_y, int dim_z, double delta_x, double delta_y, double delta_z);
	virtual ~VectorVectorConvolution_FFT();

	virtual void execute(const VectorMatrix &rhs, VectorMatrix &res);
};

%{
#include "math/gradient.h"
#include "math/ScaledAbsMax.h"
%}

void gradient(double delta_x, double delta_y, double delta_z, const Matrix &pot, VectorMatrix &field);
double scaled_abs_max(VectorMatrix &M, Matrix &scale);

