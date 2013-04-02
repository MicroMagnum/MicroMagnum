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
#include "demag_tensor.h"

#include <fstream>
#include <cstdlib> // std::getenv

#include "Logger.h"

#include "tensor.h"
#include "tensor_round.h"

#include "os.h"

//#define USE_OLD_CODE 1
#ifdef USE_OLD_CODE
#include "old/demag_old.h"
#warning "Using old code for demag tensor generation"
#endif

struct DemagTensorInfo {
	int dim_x, dim_y, dim_z; 
	int exp_x, exp_y, exp_z; 
	double delta_x, delta_y, delta_z; 
	bool periodic_x, periodic_y, periodic_z; 
	int periodic_repeat;
	int padding;
	const char *cache_dir;
};

// CACHING //////////////////////////////////////////////////////////////////////////

// Return the cache file name for a demag tensor field configuration.
static std::string cacheFile(const DemagTensorInfo &info)
{
	std::stringstream ss;
	ss << info.cache_dir << "/";
	ss << "Demag";
	ss << "--";
	ss << info.dim_x << "-" << info.dim_y << "-" << info.dim_z;
	ss << "--";
	ss << info.exp_x << "-" << info.exp_y << "-" << info.exp_z;
	ss << "--";
	ss << int(info.delta_x*1e12) << "-" << int(info.delta_y*1e12) << "-" << int(info.delta_z*1e12);
	if (info.periodic_x || info.periodic_y || info.periodic_z) {
		ss << "--";
		ss << "p-";
		ss << (info.periodic_x ? "x" : "");
		ss << (info.periodic_y ? "y" : "");
		ss << (info.periodic_z ? "z" : "");
		ss << "-" << info.periodic_repeat;
	}
	ss << ".dat";
	return ss.str();
}

static void saveDemagTensor(const DemagTensorInfo &info, const std::string &path, Matrix &mat)
{
	LOG_INFO << "Saving demagnetization tensor field to cache.";
	Matrix::ro_accessor acc(mat);
	std::ofstream out(path.c_str());
	out.write((const char*)acc.ptr(), sizeof(double) * mat.size());
	LOG_DEBUG << "Done.";
}

Matrix loadDemagTensor(const DemagTensorInfo &info, const std::string &path, bool &success /*out*/)
{
	std::ifstream in(path.c_str());
	if (in.is_open()) {
		LOG_INFO << "Loading demagnetization tensor field from cache.";

		Matrix N(Shape(6, info.exp_x, info.exp_y, info.exp_z));

		const size_t bytes_to_read = sizeof(double) * N.size(); 
		{
			Matrix::wo_accessor N_acc(N);
			in.read((char*)N_acc.ptr(), bytes_to_read);
		}
		const size_t bytes_read = in.gcount();

		if (bytes_read == bytes_to_read) {
			LOG_DEBUG << "Done.";
			success = true;
			return N;
		} else {
			LOG_ERROR << "Read error.";
			success = false;
			return Matrix(Shape());
		}
	} else {
		success = false;
		return Matrix(Shape());
	}
}

// FIELD GENERATION ////////////////////////////////////////////////////////////////////

Matrix GenerateDemagTensor(
	int dim_x, int dim_y, int dim_z, 
	double delta_x, double delta_y, double delta_z, 
	bool periodic_x, bool periodic_y, bool periodic_z, int periodic_repeat,
	int padding,
	const char *cache_dir)
{
	DemagTensorInfo info;
	info.dim_x           = dim_x; 
	info.dim_y           = dim_y; 
	info.dim_z           = dim_z;
	info.delta_x         = delta_x; 
	info.delta_y         = delta_y; 
	info.delta_z         = delta_z;
	info.periodic_x      = periodic_x; 
	info.periodic_y      = periodic_y; 
	info.periodic_z      = periodic_z;
	info.periodic_repeat = periodic_repeat;
	info.padding         = padding;
	info.cache_dir       = cache_dir;
	const int exp_x      = info.exp_x = round_tensor_dimension(dim_x, periodic_x, padding);
	const int exp_y      = info.exp_y = round_tensor_dimension(dim_y, periodic_y, padding);
	const int exp_z      = info.exp_z = round_tensor_dimension(dim_z, periodic_z, padding);

	const std::string cache_path = cacheFile(info);

	LOG_INFO << "Setting up demagnetization tensor field";
	LOG_INFO << "  Magn. size      : " << dim_x << "x" << dim_y << "x" << dim_z << " cells";
	LOG_INFO << "  FFT size        : " << exp_x << "x" << exp_y << "x" << exp_z;
	LOG_INFO << "  PBC dimensions  : " << (periodic_x ? "x" : "") << (periodic_y ? "y" : "") << (periodic_z ? "z" : "") << (!periodic_x && !periodic_y && !periodic_z ? "none" : "") << "  (" << periodic_repeat << " repetitions)";
	LOG_INFO << "  Cache file      : " << cache_path;

	// Skip computation?
	if (std::getenv("MAGNUM_DEMAG_GARBAGE")) {
		LOG_INFO << "Skipping demag tensor generation ('MAGNUM_GARBAGE' environment variable is set)";
		return Matrix(Shape(6, exp_x, exp_y, exp_z));
	}

	// Demag tensor field cached?
	{
		bool success = false;
		Matrix N_cached = loadDemagTensor(info, cache_path, success);
		if (success) return N_cached;
	}

	// periodic bc setup
	const int repeat_x = periodic_x ? periodic_repeat : 1;
	const int repeat_y = periodic_y ? periodic_repeat : 1;
	const int repeat_z = periodic_z ? periodic_repeat : 1;

	const double t0 = os::getTickCount();
#ifdef USE_OLD_CODE
	// Old implementation (in old/demag_old.cpp)
	Matrix N = calculateDemagTensor_old(delta_x, delta_y, delta_z, dim_x, dim_y, dim_z, exp_x, exp_y, exp_z, repeat_x, repeat_y, repeat_z);
#else
	// New implementation (in ./tensor.cpp)
	Matrix N = calculateDemagTensor(delta_x, delta_y, delta_z, dim_x, dim_y, dim_z, exp_x, exp_y, exp_z, repeat_x, repeat_y, repeat_z);
#endif
	const double t1 = os::getTickCount();

	// we actually compute -N because H = -(N x M) = (-N) x M
	N.scale(-1);

	// Save tensor field to cache (if the computation took longer than 30 secs)
	if (t1-t0 > 30000.0) {
		saveDemagTensor(info, cache_path, N);
	}

	// done
	return N;
}
