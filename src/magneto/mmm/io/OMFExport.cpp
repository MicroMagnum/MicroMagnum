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
#include "OMFExport.h"

#include "matrix/matty.h"

#include "endian.h"

#include <string.h>
#include <stdio.h>
#include <cfloat>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <cstddef>
#include <iostream>
#include <sstream>

#include "OMFHeader.h"

using namespace std;
using std::size_t;

static void writeAsciiValues(std::ostream &out, const VectorMatrix &field, double scale);
static void writeBinary4Values(std::ostream &out, const VectorMatrix &field, double scale);
static void writeBinary8Values(std::ostream &out, const VectorMatrix &field, double scale);
static void getMinMaxValueRange(const VectorMatrix &field, double &min, double &max);

template <class T>
std::string hdr(std::string key, T val)
{
	std::stringstream ss;
	ss.precision(16);
	ss << "# " << key << ": " << val;
	return ss.str();
}

void writeOMF(const std::string &path, OMFHeader &header, const VectorMatrix &field, OMFFormat format)
{
	std::ofstream out(path.c_str());
	if (!out.good()) {
		throw std::runtime_error(std::string("writeOMF: Could not open ") + path + " for omf file writing");
	}
	writeOMF(out, header, field, format);
}

void writeOMF(std::ostream &out, OMFHeader &header, const VectorMatrix &field, OMFFormat format)
{
	double scale = 1.0;
	double minLen, maxLen;
	getMinMaxValueRange(field, minLen, maxLen);

	bool do_scale = true;
	if (std::getenv("MAGNUM_OMF_NOSCALE")) do_scale = false;

	if (do_scale && maxLen > 0.001) {
		scale /= maxLen;
		maxLen *= scale;
		minLen *= scale;
	}

	// Adjust header values
	header.valuemultiplier = 1.0 / scale;
	header.ValueRangeMaxMag = maxLen; 
	header.ValueRangeMinMag = minLen;

	out << hdr("OOMMF", "rectangular mesh v1.0") << endl;
	out << hdr("Segment count", 1) << endl;
	out << hdr("Begin", "Segment") << endl;
	out << hdr("Begin", "Header") << endl;
	out << hdr("Title", header.Title) << endl;
	for (size_t i=0; i<header.Desc.size(); ++i) out << hdr("Desc", header.Desc[i]) << endl;
	out << hdr("meshunit", header.meshunit) << endl;
	out << hdr("valueunit", header.valueunit) << endl;
	out << hdr("valuemultiplier", header.valuemultiplier) << endl;
	out << hdr("xmin", header.xmin) << endl;
	out << hdr("ymin", header.ymin) << endl;
	out << hdr("zmin", header.zmin) << endl;
	out << hdr("xmax", header.xmax) << endl;
	out << hdr("ymax", header.ymax) << endl;
	out << hdr("zmax", header.zmax) << endl;
	out << hdr("ValueRangeMaxMag", maxLen) << endl;
	out << hdr("ValueRangeMinMag", minLen) << endl;
	out << hdr("meshtype", header.meshtype) << endl;
	out << hdr("xbase", header.xbase) << endl;
	out << hdr("ybase", header.ybase) << endl;
	out << hdr("zbase", header.zbase) << endl;
	out << hdr("xstepsize", header.xstepsize) << endl;
	out << hdr("ystepsize", header.ystepsize) << endl;
	out << hdr("zstepsize", header.zstepsize) << endl;
	out << hdr("xnodes", header.xnodes) << endl;
	out << hdr("ynodes", header.ynodes) << endl;
	out << hdr("znodes", header.znodes) << endl;
	out << hdr("End", "Header") << endl;

	switch (format) {
		case OMF_FORMAT_ASCII:
			out << hdr("Begin", "Data Text") << endl;
			writeAsciiValues(out, field, scale);
			out << hdr("End", "Data Text") << endl;
			break;

		case OMF_FORMAT_BINARY_4:
			out << hdr("Begin", "Data Binary 4") << endl;
			writeBinary4Values(out, field, scale);
			out << hdr("End", "Data Binary 4") << endl;
			break;

		case OMF_FORMAT_BINARY_8:
			out << hdr("Begin", "Data Binary 8") << endl;
			writeBinary8Values(out, field, scale);
			out << hdr("End", "Data Binary 8") << endl;
			break;
	}

	out << hdr("End", "Segment") << endl;
}

static void writeAsciiValues(std::ostream &out, const VectorMatrix &field, double scale)
{
	out.unsetf(ios_base::scientific);
	out.unsetf(ios_base::fixed);
	out.precision(16); // print the significant 52 digital digits (which is ~16 decimal digits)

	VectorMatrix::const_accessor field_acc(field);
	for (int i=0; i<field.size(); ++i) {
		const Vector3d vec = field_acc.get(i) * scale;
		out << vec.x << " " << vec.y << " " << vec.z << '\n'; // <-- no std::endl because endl implies flushing (= slowdown)
	}
}

static void writeBinary4Values(std::ostream &out, const VectorMatrix &field, double scale)
{
	assert(sizeof(float) == 4);

	VectorMatrix::const_accessor field_acc(field);

	const int num_cells = field.size();
	float *buffer = new float [3*num_cells];
	for (int i=0; i<num_cells; ++i) {
		Vector3d vec = field_acc.get(i) * scale;
		buffer[i*3+0] = toBigEndian(static_cast<float>(vec.x));
		buffer[i*3+1] = toBigEndian(static_cast<float>(vec.y));
		buffer[i*3+2] = toBigEndian(static_cast<float>(vec.z));
	}

	const float magic = toBigEndian<float>(1234567.0f);
	out.write((const char*)&magic, sizeof(float));
	out.write((const char*)buffer, num_cells * 3 * sizeof(float));
	out << endl;

	delete [] buffer;
}

static void writeBinary8Values(std::ostream &out, const VectorMatrix &field, double scale)
{
	assert(sizeof(double) == 8);

	VectorMatrix::const_accessor field_acc(field);

	const int num_cells = field.size();
	double *buffer = new double [3*num_cells];
	for (int i=0; i<num_cells; ++i) {
		Vector3d vec = field_acc.get(i) * scale;
		buffer[i*3+0] = vec.x;
		buffer[i*3+1] = vec.y;
		buffer[i*3+2] = vec.z;
	}

	const double magic = toBigEndian<double>(123456789012345.0);
	out.write((const char*)&magic, sizeof(double));
	out.write((const char*)buffer, num_cells * 3 * sizeof(double));
	out << endl;

	delete [] buffer;
}

static void getMinMaxValueRange(const VectorMatrix &field, double &min, double &max)
{
	max = -DBL_MAX;
	min = +DBL_MAX;

	VectorMatrix::const_accessor field_acc(field);

	const size_t total_nodes = field.size();
	for (size_t i=0; i<total_nodes; ++i) {
		const double len = field_acc.get(i).abs();
		if (len < min) min = len;
		if (len > max) max = len;
	}
}
