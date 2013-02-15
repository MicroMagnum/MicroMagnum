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

#ifndef TRANSFORMER_CPU_H
#define TRANSFORMER_CPU_H

#include <fftw3.h>

#include "matrix/matty.h"

class Transformer_CPU
{
public:
	Transformer_CPU(int dim_x, int dim_y, int dim_z, int exp_x, int exp_y, int exp_z);
	~Transformer_CPU();

	void transform_forward_x(double *inout);
	void transform_forward_y(double *inout);
	void transform_forward_z(double *inout);
	void transform_inverse_z(double *inout);
	void transform_inverse_y(double *inout);
	void transform_inverse_x(double *inout);

private:
	const int dim_x, dim_y, dim_z;
	const int exp_x, exp_y, exp_z;

	// FFTW plan handles
	fftw_plan plan_x_r2c, plan_x_c2r;
	fftw_plan plan_y_forw, plan_y_inv;
	fftw_plan plan_z_forw, plan_z_inv;
};

#endif
