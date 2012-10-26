/*
 * Copyright 2012 by the Micromagnum authors.
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

#ifndef MATTY_EXT_FFT_H
#define MATTY_EXT_FFT_H

#include "config.h"
#include "matrix/matty.h"

#include <vector>

namespace matty_ext 
{
	void  fftn(ComplexMatrix &inout, const std::vector<int> &loop_dims_select = std::vector<int>());
	void ifftn(ComplexMatrix &inout, const std::vector<int> &loop_dims_select = std::vector<int>());
}

#endif
