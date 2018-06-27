##############################################################
###    General                                             ###
##############################################################

# general settings
cmake_minimum_required (VERSION 2.8.0)
project (Magneto)

option(ENABLE_CUDA_32 "Enable CUDA (only 32 bit support)" OFF)
option(ENABLE_CUDA_64 "Enable CUDA (both 32 and 64 bit support, needs CUDA device capability of 2.0 or higher)" OFF)

option(USE_PYTHON2 "Use Python2" OFF)
option(USE_PYTHON3 "Use Python3" OFF)

if(ENABLE_CUDA_32 OR ENABLE_CUDA_64)
  set(ENABLE_CUDA ON)
endif(ENABLE_CUDA_32 OR ENABLE_CUDA_64)

# set cmake module path
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

##############################################################
###    Libraries   TODO: check versions                    ###
##############################################################

# Boost
# (currently unused)
#find_package (Boost REQUIRED)

# FFTW
find_library(FFTW   fftw3 REQUIRED)
find_library(FFTWF  fftw3f REQUIRED)
find_library(FFTWT  fftw3_threads REQUIRED)
find_library(FFTWFT fftw3f_threads REQUIRED)
set(FFTW_LIBRARIES "${FFTW};${FFTWF};${FFTWT};${FFTWFT}")

# SWIG + Python
find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})


##############################################################
###    Find Python                                         ###
##############################################################

if(USE_PYTHON2)
  set (USE_PYTHON 2)
elseif(USE_PYTHON3)
  set (USE_PYTHON 3)
endif(USE_PYTHON2)

find_package(Python ${USE_PYTHON} REQUIRED)

# OpenMP support (optional)
#include(FindOpenMP)

##################### include directories ##################
include_directories(
  ${PYTHON_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/magneto
  ${PROJECT_SOURCE_DIR}/magneto/matrix
  ${PROJECT_BINARY_DIR}
)

##############################################################
###    Subdirectories                                      ###
##############################################################

set(MAGNETO_SRC
  magneto/Magneto.cpp
  magneto/Benchmark.cpp
  magneto/Logger.cpp

  magneto/bindings/PythonCallable.cpp
  magneto/bindings/PythonByteArray.cpp

  magneto/evolver/runge_kutta.cpp
  magneto/evolver/runge_kutta_cpu.cpp

  magneto/matrix/Shape.cpp
  magneto/matrix/matty.cpp
  magneto/matrix/device/Array.cpp
  magneto/matrix/device/Device.cpp
  magneto/matrix/device/DeviceManager.cpp
  magneto/matrix/device/cpu/CPUArray.cpp
  magneto/matrix/device/cpu/CPUDevice.cpp
  magneto/matrix/matrix/AbstractMatrix.cpp
  magneto/matrix/matrix/scalar/Matrix.cpp
  magneto/matrix/matrix/scalar/Matrix_accessor.cpp
  magneto/matrix/matrix/vector/Vector3d.cpp
  magneto/matrix/matrix/vector/VectorMatrix.cpp
  magneto/matrix/matrix/vector/VectorMatrix_accessor.cpp
  magneto/matrix/matrix/complex/ComplexMatrix.cpp
  magneto/matrix/ext/LinearInterpolate.cpp
  magneto/matrix/ext/Extremum.cpp
  magneto/matrix/ext/FFT.cpp

  magneto/mmm/anisotropy/anisotropy.cpp
  magneto/mmm/anisotropy/anisotropy_cpu.cpp
  magneto/mmm/demag/phi/demag_phi_tensor.cpp
  magneto/mmm/demag/tensor_round.cpp
  magneto/mmm/demag/demag_static.cpp
  magneto/mmm/demag/tensor.cpp
  magneto/mmm/demag/old/demag_old.cpp
  magneto/mmm/demag/demag_tensor.cpp
  magneto/mmm/exchange/exchange.cpp
  magneto/mmm/exchange/exchange_cpu.cpp
  magneto/mmm/io/OMFHeader.cpp
  magneto/mmm/io/OMFExport.cpp
  magneto/mmm/io/OMFImport.cpp
  magneto/mmm/llge/llge.cpp
  magneto/mmm/llge/llge_cpu.cpp
  magneto/mmm/spintorque/fdm_zhangli.cpp
  magneto/mmm/spintorque/fdm_zhangli_cpu.cpp
  magneto/mmm/spintorque/fdm_slonchewski.cpp

  magneto/math/gradient.cpp
  magneto/math/ScaledAbsMax.cpp
  magneto/math/conv/MatrixVectorConvolution_FFT.cpp
  magneto/math/conv/SymmetricMatrixVectorConvolution_Simple.cpp
  magneto/math/conv/SymmetricMatrixVectorConvolution_FFT.cpp
  magneto/math/conv/AntisymmetricMatrixVectorConvolution_FFT.cpp
  magneto/math/conv/TensorFieldSetup.cpp
  magneto/math/conv/VectorVectorConvolution_FFT.cpp
  magneto/math/conv/kernels/cpu_copy_pad.cpp
  magneto/math/conv/kernels/cpu_copy_unpad.cpp
  magneto/math/conv/kernels/cpu_multiplication.cpp
  magneto/math/conv/kernels/cpu_transpose_unpad.cpp
  magneto/math/conv/kernels/cpu_transpose_zeropad.cpp
  magneto/math/conv/kernels/Transposer_CPU.cpp
  magneto/math/conv/kernels/Transformer_CPU.cpp
)

# Add Cuda specific sources
if(ENABLE_CUDA)
  set(MAGNETO_SRC ${MAGNETO_SRC}
    magneto/matrix/device/cuda/CUDADevice.cpp
    magneto/matrix/device/cuda/CUDAArray.cpp
    magneto/matrix/device/cuda_tools.cpp
    magneto/matrix/matrix/scalar/Matrix_cuda_accessor.cpp
    magneto/matrix/matrix/vector/VectorMatrix_cuda_accessor.cpp

    magneto/mesh/VectorField.cpp
    magneto/mesh/Field.cpp

    magneto/math/gradient_cuda.cu
    magneto/math/ScaledAbsMax_cuda.cu
    magneto/math/conv/kernels/cuda_transpose_unpad.cpp
    magneto/math/conv/kernels/cuda_transpose_zeropad.cpp
    magneto/math/conv/kernels/Transformer_CUDA.cpp
    magneto/math/conv/kernels/Transposer_CUDA.cpp
  )
endif(ENABLE_CUDA)

# Add cuda-only sources
set(MAGNETO_CUSRC
  magneto/evolver/runge_kutta_cuda.cu

  magneto/matrix/device/cuda/kernels_reduce.cu
  magneto/matrix/device/cuda/kernels_simple.cu

  magneto/mmm/anisotropy/anisotropy_cuda.cu
  magneto/mmm/exchange/exchange_cuda.cu
  magneto/mmm/llge/llge_cuda.cu
  magneto/mmm/spintorque/fdm_zhangli_cuda.cu

  magneto/math/conv/kernels/cuda_copy_pad.cu
  magneto/math/conv/kernels/cuda_copy_unpad.cu
  magneto/math/conv/kernels/cuda_multiplication.cu
  magneto/math/conv/kernels/cuda_transpose_unpad_2d.cu
  magneto/math/conv/kernels/cuda_transpose_unpad_3d.cu
  magneto/math/conv/kernels/cuda_transpose_zeropad_2d.cu
  magneto/math/conv/kernels/cuda_transpose_zeropad_3d.cu
)

# Add operating system specific sources
if(UNIX)
  set(MAGNETO_SRC ${MAGNETO_SRC} "magneto/os_posix.cpp")
elseif(WINDOWS)
  set(MAGNETO_SRC ${MAGNETO_SRC} "magneto/os_windows.cpp")
endif()

##############################################################
###    Set Compiler Flags                                  ###
##############################################################

if(CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -march=native -msse3 -fomit-frame-pointer -ffast-math -fpermissive -fPIC")
endif(CMAKE_COMPILER_IS_GNUCXX)

if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

##############################################################
###    Compile                                             ###
##############################################################

# Compile Cuda kernels into library
if(ENABLE_CUDA)
  include(FindCUDA)
  #find_package(CUDA)

  # Set HAVE_CUDA and HAVE_CUDA_64 (for config.h)
  set(HAVE_CUDA ON)
  if(ENABLE_CUDA_64)
    set(HAVE_CUDA_64 ON)
  endif(ENABLE_CUDA_64)

  # Setup nvcc command line args
  set(CUDA_PROPAGATE_HOST_FLAGS OFF)

  if(ENABLE_CUDA_64)
    set(CUDA_NVCC_FLAGS --gpu-architecture=compute_20;-Xcompiler;-fpic;-O6;--use_fast_math)
  else(ENABLE_CUDA_64)
    set(CUDA_NVCC_FLAGS -Xcompiler;-fpic;-O6;--use_fast_math)
  endif(ENABLE_CUDA_64)

  cuda_add_library(magneto_cu_parts ${MAGNETO_CUSRC})
  cuda_add_cufft_to_target(magneto_cu_parts)
  cuda_add_cublas_to_target(magneto_cu_parts)
endif(ENABLE_CUDA)

# Compile Swig module
set(BINDINGS_SOURCE ${PROJECT_SOURCE_DIR}/magneto/bindings/magneto.i)
set_source_files_properties (${BINDINGS_SOURCE} PROPERTIES CPLUSPLUS ON )
if(ENABLE_CUDA)
  set(SWIG_MODULE_NAME "magneto_cuda")
else()
  set(SWIG_MODULE_NAME "magneto_cpu")
endif(ENABLE_CUDA)
set_source_files_properties (${BINDINGS_SOURCE} PROPERTIES SWIG_FLAGS "-O;-module;${SWIG_MODULE_NAME}")
swig_add_module(${SWIG_MODULE_NAME} python ${PROJECT_SOURCE_DIR}/magneto/bindings/magneto.i ${MAGNETO_SRC})
swig_link_libraries(${SWIG_MODULE_NAME} ${FFTW_LIBRARIES})

# Link Cuda Kernels to Swig module
if(ENABLE_CUDA)
  swig_link_libraries(${SWIG_MODULE_NAME} magneto_cu_parts)
endif(ENABLE_CUDA)


##############################################################
###    Create a config.h                                   ###
##############################################################

# Add configuration file config.h
configure_file(
  "${PROJECT_SOURCE_DIR}/magneto/config.h.in"
  "${PROJECT_BINARY_DIR}/config.h"
)

##############################################################
###    Install rule                                        ###
##############################################################

# set PYTHON_SITE_MODULES path
message ("-- Found Python ${PYTHON_VERSION}, will install at ${PYTHON_SITE_MODULES}")

# install directory magnum, exclude the symlinks
install(DIRECTORY magnum DESTINATION "${PYTHON_SITE_MODULES}" PATTERN "magnum/magneto_cuda.py magnum/_magneto_cuda.so magnum/magneto_cpu.py magnum/_magneto_cpu.so CMakeLists.txt" EXCLUDE)

# copy cuda files, if used
if(ENABLE_CUDA)
  install(FILES build/magneto_cuda.py build/_magneto_cuda.so DESTINATION "${PYTHON_SITE_MODULES}/magnum")
else(ENABLE_CUDA)
  # copy built destinations of cpu symlinks
  install(FILES build/magneto_cpu.py build/_magneto_cpu.so DESTINATION "${PYTHON_SITE_MODULES}/magnum")
endif(ENABLE_CUDA)

