//%module xyz // this is now set via the command line

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
