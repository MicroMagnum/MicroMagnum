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

#ifndef TRANSPOSER_CPU_H
#define TRANSPOSER_CPU_H

#include "matrix/matty.h"

#include <fftw3.h>

class Transposer_CPU
{
public:
	Transposer_CPU(int dim_x, int dim_y, int dim_z, int exp_x, int exp_y, int exp_z);
	~Transposer_CPU();

	void copy_pad(const VectorMatrix &M, double *out_x, double *out_y, double *out_z);
	void transpose_zeropad_yzx(const double *in_x, const double *in_y, const double *in_z, double *out_x, double *out_y, double *out_z);
	void transpose_zeropad_zxy(const double *in_x, const double *in_y, const double *in_z, double *out_x, double *out_y, double *out_z);
	void transpose_unpad_yzx(const double *in_x, const double *in_y, const double *in_z, double *out_x, double *out_y, double *out_z);
	void transpose_unpad_xyz(const double *in_x, const double *in_y, const double *in_z, double *out_x, double *out_y, double *out_z);
	void copy_unpad(const double *in_x, const double *in_y, const double *in_z, VectorMatrix &H);

private:
	const int dim_x, dim_y, dim_z;
	const int exp_x, exp_y, exp_z;

	fftw_plan plan_unpad_zxy_yzx, plan_unpad_yzx_xyz;

	void initPlans();
	void deinitPlans();
};

#endif
