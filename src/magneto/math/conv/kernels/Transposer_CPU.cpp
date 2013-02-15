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

#include "Transposer_CPU.h"

#include "cpu_copy_pad.h"
#include "cpu_copy_unpad.h"
#include "cpu_transpose_zeropad.h"
#include "cpu_transpose_unpad.h"

Transposer_CPU::Transposer_CPU(int dim_x, int dim_y, int dim_z, int exp_x, int exp_y, int exp_z)
	: dim_x(dim_x), dim_y(dim_y), dim_z(dim_z), exp_x(exp_x), exp_y(exp_y), exp_z(exp_z)
{
	initPlans();
}

Transposer_CPU::~Transposer_CPU()
{
	deinitPlans();
}

void Transposer_CPU::initPlans()
{
	fftw_iodim loop[3];

	double *in  = new double [2*(exp_x/2+1)*exp_y*dim_z];
	double *out = new double [2*(exp_x/2+1)*dim_y*dim_z]; 

	// yzx->xyz
	/*cpu_transpose_unpad_c2c(
		exp_y, dim_z, exp_x/2+1,
		dim_y, 
		in_x, in_y, in_z,
		out_x, out_y, out_z
	);*/
	loop[0].n  = dim_y;
	loop[0].is = 1;
	loop[0].os = exp_x/2+1;
	loop[1].n  = dim_z;
	loop[1].is = exp_y;
	loop[1].os = (exp_x/2+1) * dim_y;
	loop[2].n  = exp_x/2+1;
	loop[2].is = exp_y * dim_z;
	loop[2].os = 1;

	plan_unpad_yzx_xyz = fftw_plan_guru_dft(
		0, NULL,
		3, loop,
		(fftw_complex*)in, 
		(fftw_complex*)out, 
		FFTW_FORWARD,
		FFTW_MEASURE
	);

	delete [] in;
	delete [] out;
	
	//plan_unpad_zxy_yzx
}

void Transposer_CPU::deinitPlans()
{
	fftw_destroy_plan(plan_unpad_yzx_xyz);
}

void Transposer_CPU::copy_pad(const VectorMatrix &M, double *out_x, double *out_y, double *out_z)
{
	VectorMatrix::const_accessor M_acc(M);
	const double *in_x = M_acc.ptr_x();
	const double *in_y = M_acc.ptr_y();
	const double *in_z = M_acc.ptr_z();

	cpu_copy_pad_r2r(
		dim_x, dim_y, dim_z, 
		exp_x, 
		in_x, in_y, in_z,
		out_x, out_y, out_z
	);
}

void Transposer_CPU::copy_unpad(const double *in_x, const double *in_y, const double *in_z, VectorMatrix &H)
{
	VectorMatrix::accessor H_acc(H);
	double *out_x = H_acc.ptr_x();
	double *out_y = H_acc.ptr_y();
	double *out_z = H_acc.ptr_z();

	cpu_copy_unpad_r2r(
		exp_x, dim_y, dim_z, 
		dim_x, 
		in_x, in_y, in_z,
		out_x, out_y, out_z
	);
}

void Transposer_CPU::transpose_zeropad_yzx(const double *in_x, const double *in_y, const double *in_z, double *out_x, double *out_y, double *out_z)
{
	// xyz->yzx
	cpu_transpose_zeropad_c2c(
		exp_x/2+1, dim_y, dim_z, 
		exp_y, 
		in_x, in_y, in_z,
		out_x, out_y, out_z
	);
}

void Transposer_CPU::transpose_zeropad_zxy(const double *in_x, const double *in_y, const double *in_z, double *out_x, double *out_y, double *out_z)
{
	// yzx->zxy
	cpu_transpose_zeropad_c2c(
		exp_y, dim_z, exp_x/2+1,
		exp_z, 
		in_x, in_y, in_z,
		out_x, out_y, out_z
	);
}

void Transposer_CPU::transpose_unpad_yzx(const double *in_x, const double *in_y, const double *in_z, double *out_x, double *out_y, double *out_z)
{
	// zxy->yzx
	cpu_transpose_unpad_c2c(
		exp_z, exp_x/2+1, exp_y,
		dim_z, 
		in_x, in_y, in_z,
		out_x, out_y, out_z
	);
}

void Transposer_CPU::transpose_unpad_xyz(const double *in_x, const double *in_y, const double *in_z, double *out_x, double *out_y, double *out_z)
{
	// yzx->xyz
	cpu_transpose_unpad_c2c(
		exp_y, dim_z, exp_x/2+1,
		dim_y, 
		in_x, in_y, in_z,
		out_x, out_y, out_z
	);
	/*fftw_execute_dft(plan_unpad_yzx_xyz, (fftw_complex*)in_x, (fftw_complex*)out_x);
	fftw_execute_dft(plan_unpad_yzx_xyz, (fftw_complex*)in_y, (fftw_complex*)out_y);
	fftw_execute_dft(plan_unpad_yzx_xyz, (fftw_complex*)in_z, (fftw_complex*)out_z);*/
}

