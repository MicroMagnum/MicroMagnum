##############################################################
###    General                                             ###
##############################################################

cmake_minimum_required(VERSION 2.8.0)
project(Magneto)

option(ENABLE_CUDA_32 "Enable CUDA (only 32 bit support)" OFF)
option(ENABLE_CUDA_64 "Enable CUDA (both 32 and 64 bit support, needs CUDA device capability of 2.0 or higher)" OFF)

option(USE_PYTHON2 "Use Python2" OFF)
option(USE_PYTHON3 "Use Python3" OFF)

if(ENABLE_CUDA_32 OR ENABLE_CUDA_64)
  set(ENABLE_CUDA ON)
endif()

##############################################################
###    Libraries                                           ###
##############################################################

# FFTW
find_library(FFTW fftw3 REQUIRED)
find_library(FFTWT fftw3_threads REQUIRED)
set(FFTW_LIBRARIES "${FFTW};${FFTWT}")

# SWIG
find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

# CMake version 2.8.8+ will select the highest Python version including Python 3.
# CMake version 2.8.7- will always choose Python 2.x.
if(USE_PYTHON2)
  set(Python_ADDITIONAL_VERSIONS 2.8 2.7 2.6)
elseif(USE_PYTHON3)
  set(Python_ADDITIONAL_VERSIONS 3.3 3.2 3.1 3.0)
endif()

find_package(PythonLibs REQUIRED)

# OpenMP support (optional)
include(FindOpenMP)

# CUDA (if enabled)
if(ENABLE_CUDA)
  include(FindCUDA)
  #find_package(CUDA)

  # Set HAVE_CUDA and HAVE_CUDA_64 (for config.h)
  set(HAVE_CUDA ON)
  if(ENABLE_CUDA_64)
    set(HAVE_CUDA_64 ON)
  endif()
endif()

##############################################################
###    Subdirectories                                      ###
##############################################################

# Macro for adding src files from within subdirectories
macro(append_sources DIR CPP_FILES CPP_FILES_FOR_CUDA CU_FILES)
  set(LIST)
  foreach(FILE ${CPP_FILES})
    list(APPEND LIST ${DIR}/${FILE})
  endforeach()
  set(MAGNETO_CPU_SRC ${MAGNETO_CPU_SRC};${LIST} PARENT_SCOPE)
  set(MAGNETO_CUDA_SRC ${MAGNETO_CUDA_SRC};${LIST} PARENT_SCOPE)

  set(LIST)
  foreach(FILE ${CPP_FILES_FOR_CUDA})
    list(APPEND LIST ${DIR}/${FILE})
  endforeach()
  set(MAGNETO_CUDA_SRC ${MAGNETO_CUDA_SRC};${LIST} PARENT_SCOPE)

  set(LIST)
  foreach(FILE ${CU_FILES})
    list(APPEND LIST ${DIR}/${FILE})
  endforeach()
  set(MAGNETO_CUDA_CUSRC ${MAGNETO_CUDA_CUSRC};${LIST} PARENT_SCOPE)
endmacro()

add_subdirectory(magneto)
add_subdirectory(magneto/bindings)
add_subdirectory(magneto/matrix)
add_subdirectory(magneto/math)
add_subdirectory(magneto/math/conv)
add_subdirectory(magneto/math/conv/kernels)
add_subdirectory(magneto/mmm)
add_subdirectory(magneto/evolver)
add_subdirectory(magneto/mesh)

##############################################################
###    Set Compiler Flags                                  ###
##############################################################

if(CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -march=native -msse3 -fomit-frame-pointer -ffast-math")
endif()

if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

if(ENABLE_CUDA)
  set(CUDA_PROPAGATE_HOST_FLAGS OFF)

  if(ENABLE_CUDA_64)
    set(CUDA_NVCC_FLAGS --gpu-architecture=compute_20;-Xcompiler;-fpic;-O6;--use_fast_math)
  else(ENABLE_CUDA_64)
    set(CUDA_NVCC_FLAGS -Xcompiler;-fpic;-O6;--use_fast_math)
  endif()
endif()

include_directories(
  ${PYTHON_INCLUDE_PATH}
  ${PROJECT_SOURCE_DIR}/magneto
  ${PROJECT_SOURCE_DIR}/magneto/matrix
  ${PROJECT_BINARY_DIR}
)

##############################################################
###    Compile                                             ###
##############################################################

set(BINDINGS_CPU_SOURCE ${PROJECT_SOURCE_DIR}/magneto/bindings/magneto_cpu.i)
set(BINDINGS_CUDA_SOURCE ${PROJECT_SOURCE_DIR}/magneto/bindings/magneto_cuda.i)
set_source_files_properties(${BINDINGS_CPU_SOURCE} ${BINDINGS_CUDA_SOURCE} PROPERTIES CPLUSPLUS ON)
set_source_files_properties(${BINDINGS_CPU_SOURCE} ${BINDINGS_CUDA_SOURCE} PROPERTIES SWIG_FLAGS "-O;-threads")

# magneto_cpu module
swig_add_module(magneto_cpu python ${BINDINGS_CPU_SOURCE} ${MAGNETO_CPU_SRC})
swig_link_libraries(magneto_cpu ${FFTW_LIBRARIES})

# magneto_cuda module (if enabled)
if(ENABLE_CUDA)
  cuda_add_library(magneto_cu_parts ${MAGNETO_CUDA_CUSRC})
  cuda_add_cufft_to_target(magneto_cu_parts)
  cuda_add_cublas_to_target(magneto_cu_parts)

  swig_add_module(magneto_cuda python ${PBINDINGS_CUDA_SOURCE} ${MAGNETO_CUDA_SRC})
  swig_link_libraries(magneto_cuda ${FFTW_LIBRARIES} magneto_cu_parts)
endif()

##############################################################
###    Don't forget to create config.h                     ###
##############################################################

# Add configuration file config.h
configure_file(
  "${PROJECT_SOURCE_DIR}/magneto/config.cmake-generated.h.in"
  "${PROJECT_BINARY_DIR}/config.cmake-generated.h"
)

##############################################################
###    Install rule                                        ###
##############################################################

#message("${PYTHONLIBS_VERSION_STRING}")
STRING(SUBSTRING "${PYTHONLIBS_VERSION_STRING}" 0 3 PYTHON_MAJOR)

# set PYTHON_SITE_PACKAGES path
execute_process ( COMMAND python${PYTHON_MAJOR} -c "import sys; from distutils.sysconfig import get_python_lib; d0 = get_python_lib(0,0,'/usr/local'); d1 = get_python_lib(); print(d0 if d0 in sys.path else d1)" OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)
message ("-- Found Python ${PYTHONLIBS_VERSION_STRING}, will install at ${PYTHON_SITE_PACKAGES}")

# install directory magnum, exclude the symlinks
install(DIRECTORY magnum DESTINATION "${PYTHON_SITE_PACKAGES}" PATTERN "magnum/magneto_cuda.py magnum/_magneto_cuda.so magnum/magneto_cpu.py magnum/_magneto_cpu.so CMakeLists.txt" EXCLUDE)

# copy cuda files, if used
if(ENABLE_CUDA)
  install(FILES build/magneto_cuda.py build/_magneto_cuda.so DESTINATION "${PYTHON_SITE_PACKAGES}/magnum")
else()
  # copy builded destinations of cpu symlinks
  install(FILES build/magneto_cpu.py build/_magneto_cpu.so DESTINATION "${PYTHON_SITE_PACKAGES}/magnum")
endif()
