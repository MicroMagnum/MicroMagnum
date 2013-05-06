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
#include "FFT.h"

#include <fftw3.h>
#include <cassert>
#include <algorithm>

#include "os.h"

namespace matty_ext {

static void make_iodims(
	const Shape &shape,
	const std::vector<int> &loop_dims_select,
	int &num_dims, fftw_iodim *dims, 
	int &num_loop_dims, fftw_iodim *loop_dims)
{
	const int rank = shape.getRank();

	num_dims = 0;
	num_loop_dims = 0;

	// Create loop dims
	for (int r=0; r<rank; ++r) {
		// Is loop or transform dim?
		const bool is_loop_dim = (std::find(loop_dims_select.begin(), loop_dims_select.end(), r) != loop_dims_select.end());
		fftw_iodim *dim = 0;
		if (is_loop_dim) {
			dim = &loop_dims[num_loop_dims++];
		} else {
			dim = &dims[num_dims++];
		}

		// Fill in iodims
		dim->n  = shape.getDim(r);
		dim->is = shape.getStride(r);
		dim->os = shape.getStride(r);
	}

	assert(num_loop_dims + num_dims == rank);
}

static void transform(ComplexMatrix &inout, const std::vector<int> &loop_dims_select, unsigned fftw_flags, bool forward)
{
	if (os::disable_SSE_for_FFTW()) {
		fftw_flags |= FFTW_UNALIGNED; // see os.h
	}

	int num_dims, num_loop_dims;
	fftw_iodim dims[16], loop_dims[16];
	make_iodims(inout.getShape(), loop_dims_select, num_dims, dims, num_loop_dims, loop_dims);

	ComplexMatrix::accessor inout_acc(inout);
	double *re = inout_acc.ptr_real();
	double *im = inout_acc.ptr_imag();

	if (!forward) {
		std::swap(re, im);
	}

	fftw_plan plan = fftw_plan_guru_split_dft(
		num_dims, dims, 
		num_loop_dims, loop_dims,
		re, im, // in
		re, im, // out
		fftw_flags
	);
	assert(plan);

	fftw_execute_split_dft(plan, re, im, re, im);

	fftw_destroy_plan(plan);
}

void fftn(ComplexMatrix &inout, const std::vector<int> &loop_dims_select)
{
	transform(inout, loop_dims_select, FFTW_ESTIMATE, true);
}

void ifftn(ComplexMatrix &inout, const std::vector<int> &loop_dims_select)
{
	transform(inout, loop_dims_select, FFTW_ESTIMATE, false);
}

} // ns
