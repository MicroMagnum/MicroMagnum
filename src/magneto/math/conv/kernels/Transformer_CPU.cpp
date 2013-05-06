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

#include "Transformer_CPU.h"

#include <stdexcept>
#include <cassert>

#include "os.h"

static int fftw_strategy = FFTW_MEASURE;

Transformer_CPU::Transformer_CPU(int dim_x, int dim_y, int dim_z, int exp_x, int exp_y, int exp_z)
	: dim_x(dim_x), dim_y(dim_y), dim_z(dim_z), exp_x(exp_x), exp_y(exp_y), exp_z(exp_z)
{
	if (os::disable_SSE_for_FFTW()) {
		fftw_strategy |= FFTW_UNALIGNED; // see os.h for explanation
	}

	Matrix tmp(Shape(2, exp_x, exp_y, exp_z));
	Matrix::rw_accessor tmp_acc(tmp);
	double *tmp_inout = tmp_acc.ptr();

	// Create fftw plans
	fftw_iodim dims, loop;

	// X-Transform: (dim_y*dim_z) x 1d-C2C-FFT (length: exp_x) in x-direction, in-place transform
	dims.n = exp_x;
	dims.is = 1;
	dims.os = 1;
	
	loop.n = dim_y*dim_z;
	loop.is = exp_x;
	loop.os = exp_x/2+1;

	plan_x_r2c = fftw_plan_guru_dft_r2c(
		1, &dims, 
		1, &loop, 
		(      double*)tmp_inout,
		(fftw_complex*)tmp_inout, 
		fftw_strategy
	);
	assert(plan_x_r2c);

	dims.n = exp_x;
	dims.is = 1;
	dims.os = 1;
	
	loop.n = dim_y*dim_z;
	loop.is = exp_x/2+1;
	loop.os = exp_x;

	plan_x_c2r = fftw_plan_guru_dft_c2r(
		1, &dims, 
		1, &loop, 
		(fftw_complex*)tmp_inout, 
		(      double*)tmp_inout,
		fftw_strategy
	);
	assert(plan_x_c2r);

	// Y-Transform: (dim_z*exp_x/2+1) x 1d-C2C-FFT (length: exp_y) in x-direction, in-place transform
	dims.n = exp_y;
	dims.is = 1;
	dims.os = 1;
	
	loop.n = dim_z*(exp_x/2+1);
	loop.is = exp_y;
	loop.os = exp_y;

	plan_y_forw = fftw_plan_guru_dft(
		1, &dims,
		1, &loop,
		(fftw_complex*)tmp_inout, // in
		(fftw_complex*)tmp_inout, // out (-> in-place transform)
		FFTW_FORWARD,
		fftw_strategy
	);
	assert(plan_y_forw);

	plan_y_inv = fftw_plan_guru_dft(
		1, &dims,
		1, &loop,
		(fftw_complex*)tmp_inout, // in
		(fftw_complex*)tmp_inout, // out (-> in-place transform)
		FFTW_BACKWARD,
		fftw_strategy
	);
	assert(plan_y_inv);

	// Z-Transform: (exp_x/2+1*exp_y) x 1d-C2C-FFT (length: exp_z) in x-direction, in-place transform
	dims.n = exp_z;
	dims.is = 1;
	dims.os = 1;
	
	loop.n = (exp_x/2+1)*exp_y;
	loop.is = exp_z;
	loop.os = exp_z;

	plan_z_forw = fftw_plan_guru_dft(
		1, &dims,
		1, &loop,
		(fftw_complex*)tmp_inout, // in
		(fftw_complex*)tmp_inout, // out (-> in-place transform)
		FFTW_FORWARD,
		fftw_strategy
	);
	assert(plan_z_forw);

	plan_z_inv = fftw_plan_guru_dft(
		1, &dims,
		1, &loop,
		(fftw_complex*)tmp_inout, // in
		(fftw_complex*)tmp_inout, // out (-> in-place transform)
		FFTW_BACKWARD,
		fftw_strategy
	);
	assert(plan_z_inv);
}

Transformer_CPU::~Transformer_CPU()
{
	fftw_destroy_plan(plan_x_r2c);
	fftw_destroy_plan(plan_x_c2r);
	fftw_destroy_plan(plan_y_forw);
	fftw_destroy_plan(plan_y_inv);
	fftw_destroy_plan(plan_z_forw);
	fftw_destroy_plan(plan_z_inv);
}

void Transformer_CPU::transform_forward_x(double *inout)
{
	fftw_execute_dft_r2c(plan_x_r2c, (double*)inout, (fftw_complex*)inout);
}

void Transformer_CPU::transform_forward_y(double *inout)
{
	fftw_execute_dft(plan_y_forw, (fftw_complex*)inout, (fftw_complex*)inout);
}

void Transformer_CPU::transform_forward_z(double *inout)
{
	fftw_execute_dft(plan_z_forw, (fftw_complex*)inout, (fftw_complex*)inout);
}

void Transformer_CPU::transform_inverse_z(double *inout)
{
	fftw_execute_dft(plan_z_inv, (fftw_complex*)inout, (fftw_complex*)inout);
}

void Transformer_CPU::transform_inverse_y(double *inout)
{
	fftw_execute_dft(plan_y_inv, (fftw_complex*)inout, (fftw_complex*)inout);
}

void Transformer_CPU::transform_inverse_x(double *inout)
{
	fftw_execute_dft_c2r(plan_x_c2r, (fftw_complex*)inout, (double*)inout);
}
