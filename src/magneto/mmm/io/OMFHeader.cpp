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
#include "OMFHeader.h"

OMFHeader::OMFHeader()
	: Title("<title>"),
	  meshunit("<meshunit>"),
	  valueunit("<valueunit>"),
	  valuemultiplier(1.0),
	  xmin(0.0), ymin(0.0), zmin(0.0),
	  xmax(0.0), ymax(0.0), zmax(0.0),
	  ValueRangeMaxMag(0.0), ValueRangeMinMag(0.0),
	  meshtype("rectangular"),
	  xbase(0.0), ybase(0.0), zbase(0.0),
	  xstepsize(0.0), ystepsize(0.0), zstepsize(0.0),
	  xnodes(0), ynodes(0), znodes(0)
{
}

OMFHeader::~OMFHeader()
{
}
