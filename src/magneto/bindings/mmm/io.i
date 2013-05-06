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
#include "mmm/io/OMFExport.h"
#include "mmm/io/OMFImport.h"
#include "mmm/io/OMFHeader.h"
%}

enum OMFFormat 
{
	OMF_FORMAT_ASCII=0,
	OMF_FORMAT_BINARY_4=1,
	OMF_FORMAT_BINARY_8=2
};

struct OMFHeader 
{
	OMFHeader();
	~OMFHeader();

	std::string Title;
	std::vector<std::string> Desc;
	std::string meshunit;
	std::string valueunit;
	double valuemultiplier;
	double xmin, ymin, zmin;
	double xmax, ymax, zmax;
	double ValueRangeMaxMag, ValueRangeMinMag;
	std::string meshtype;
	double xbase, ybase, zbase;
	double xstepsize, ystepsize, zstepsize;
	int xnodes, ynodes, znodes;
};

VectorMatrix  readOMF(const std::string &path, OMFHeader &header);
void         writeOMF(const std::string &path, OMFHeader &header, const VectorMatrix &field, OMFFormat format);

