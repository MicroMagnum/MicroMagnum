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
