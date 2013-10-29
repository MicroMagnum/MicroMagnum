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

#include "bindings/PythonCallable.h"
#include "os.h"

#include "Magneto.h"
#include "Benchmark.h"
#include "Logger.h"

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#include "matrix/matty.h"
#endif

#include <fftw3.h>

#include <cstdlib> // std::free
#include <fstream>
#include <iostream>
using namespace std;

// Global settings //
static PythonCallable callback;
static CudaMode cuda_mode = CUDA_DISABLED;
static bool profiling_enabled = false;

void setDebugCallback(PythonCallable &callback)
{
	::callback = callback;
}

void callDebugFunction(int level, const std::string &msg)
{
	callback.call(level, msg);
}

void enableCuda(CudaMode mode, int gpu_id)
{
	cuda_mode = mode;

	if (mode == CUDA_32 || mode == CUDA_64) {
#ifdef HAVE_CUDA
		LOG_DEBUG << "GPU is enabled (" << (mode == CUDA_32 ? "32" : "64") << " bit mode)";
		::initialize_cuda(gpu_id);
		matty::getDeviceManager().addCUDADevice(gpu_id);
#else
		cuda_mode = CUDA_DISABLED;
		LOG_ERROR << "Can't enable GPU because GPU support is not available (not enabled at compile time).";
		LOG_DEBUG << "GPU is disabled!";
#endif
	} else {
		LOG_DEBUG << "GPU is disabled!";
	}
}

bool isCudaEnabled()
{
#ifdef HAVE_CUDA
	return cuda_mode != CUDA_DISABLED;
#else
	return false;
#endif
}

bool isCuda64Enabled()
{
#ifdef HAVE_CUDA_64
	return cuda_mode == CUDA_64;
#else
	return false;
#endif
}

bool haveCudaSupport()
{
#ifdef HAVE_CUDA
	return true;
#else
	return false;
#endif
}

void cudaSync()
{
#ifdef HAVE_CUDA
	if (isCudaEnabled()) {
		checkCudaSuccess(cudaThreadSynchronize());
	}
#endif
}

void enableProfiling(bool yes)
{
	::profiling_enabled = yes;
}

bool isProfilingEnabled()
{
	return ::profiling_enabled;
}

void printProfilingReport()
{
	if (isProfilingEnabled()) {
		Logger l(Logger::LOG_LEVEL_INFO);
		l.log(__FILE__, __LINE__) << "Profiling data:" << std::endl;
		Benchmark::inst().report(l.log(__FILE__, __LINE__));
	} else {
		LOG_ERROR << "Can't print profiling report (profiling not enabled)";
	}
}

bool haveFFTWThreads()
{
#ifdef HAVE_FFTW_THREADS
	return true;
#else
	return false;
#endif
}

void setFFTWThreads(int num_threads)
{
#ifdef HAVE_FFTW_THREADS
	LOG_INFO << "FFTW using " << num_threads << " threads from now on";
	fftw_plan_with_nthreads(num_threads);
#else
	LOG_WARN << "FFTW thread support not compiled in: FFTs are single-threaded";
#endif
}

static const std::string fftw_wisdom_file = "fftw.wisdom";

void initialize(const std::string &config_path)
{
#ifdef HAVE_FFTW_THREADS
	// Enable FFTW threads
	fftw_init_threads();
#endif

	// Load FFTW wisdom if any
	const std::string fftw_wisdom_path = config_path + os::pathSeparator() + fftw_wisdom_file;
	std::ifstream f(fftw_wisdom_path.c_str());
	if (f) {
		// read f into string
		std::stringstream ss; std::string line;
		while (std::getline(f, line)) ss << line << "\n";
		// and import wisdom from string
		if (fftw_import_wisdom_from_string(ss.str().c_str())) {
			LOG_DEBUG << "Imported FFTW wisdom from file";
		} else {
			LOG_ERROR << "FFTW wisdom file seems to be invalid.";
		}
	} else {
		LOG_INFO << "Failed to import FFTW wisdom from file " << fftw_wisdom_path;
	}
}

void deinitialize(const std::string &config_path)
{
	// Save FFTW wisdom
	const std::string fftw_wisdom_path = config_path + os::pathSeparator() + fftw_wisdom_file;
	char *wisdom = fftw_export_wisdom_to_string();
	std::ofstream f(fftw_wisdom_path.c_str());
	f << wisdom;
	std::free(wisdom);

#ifdef HAVE_FFTW_THREADS
	fftw_cleanup_threads();
#endif

	setDebugCallback(PythonCallable());
}

void flush()
{
	cudaSync();
}
