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

#ifndef MAGNETO_H
#define MAGNETO_H

#include "config.h"
#include <string>

// Global initialization
void initialize(const std::string &config_path);
void deinitialize(const std::string &config_path);

// Logging functions.
void setDebugCallback(class PythonCallable &callback);
void callDebugFunction(int level, const std::string &msg);

// Profiling
void enableProfiling(bool yes);
bool isProfilingEnabled();
void printProfilingReport();

#define TIC(id) { if (isProfilingEnabled()) Benchmark::inst().tic(id); }
#define TOC(id) { if (isProfilingEnabled()) Benchmark::inst().toc(id); }

#define CUTIC(id) { if (isProfilingEnabled()) { cudaThreadSynchronize(); Benchmark::inst().tic(id); } }
#define CUTOC(id) { if (isProfilingEnabled()) { cudaThreadSynchronize(); Benchmark::inst().toc(id); } }

#ifdef HAVE_CUDA
#define SWIG_SYNCHRONIZE() do { CUDA_THREAD_SYNCHRONIZE(); } while (false);
#else
#define SWIG_SYNCHRONIZE() while (false);
#endif

// CUDA support control
enum CudaMode {
	CUDA_DISABLED=0,
	CUDA_32=1,
	CUDA_64=2
};
void enableCuda(CudaMode mode, int gpu_id = -1);
bool isCudaEnabled();
bool isCuda64Enabled();
bool haveCudaSupport();
void cudaSync();

// Set FFTW thread configuration
bool haveFFTWThreads();
void setFFTWThreads(int num_threads);

// Flush memory allocators
void flush();

#endif
