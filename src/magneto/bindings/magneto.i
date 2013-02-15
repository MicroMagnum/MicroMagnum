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

////////////////////////// C++ compatibility definitions //////////////////////////

%include "std_string.i"
%include "std_vector.i"
%include "typemaps.i"

namespace std {
    %template(StringVector) vector<string>;
    %template(IntVector) vector<int>;
}

%exception {
        try {
                $action
        } catch (std::exception &e) {
                PyErr_SetString(PyExc_RuntimeError, const_cast<char*>(e.what()));
                return 0;
        } catch (...) {
                PyErr_SetString(PyExc_RuntimeError, "Some C++ exception occured!");
                return 0;
        }
}

//////////////////////////////// Special typemaps /////////////////////////////////

%{
#include "bindings/PythonCallable.h"
#include "bindings/PythonByteArray.h"
%}

%include "PythonCallable.i"
%include "PythonByteArray.i"

///////////////////////////////////////////////////////////////////////////////////

// Magneto parts
%include "../matrix/matty.inc.i" // Matrix subsystem definitions
%include "mmm.i"
%include "math.i"
%include "evolver.i"
%include "benchmark.i"

%{
#include "Magneto.h"
%}

void initialize(const std::string &config_path);
void deinitialize(const std::string &config_path);

void setDebugCallback(PythonCallable callback);
void callDebugFunction(int level, const std::string &msg);

void enableProfiling(bool yes);
bool isProfilingEnabled();
void printProfilingReport();

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

bool haveFFTWThreads();
void setFFTWThreads(int num_threads);

void flush();
