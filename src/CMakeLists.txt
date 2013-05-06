##############################################################
###    General                                             ###
##############################################################

# general settings
cmake_minimum_required (VERSION 2.8.8)
project (Magneto)

option(ENABLE_CUDA_32 "Enable CUDA (only 32 bit support)" OFF)
option(ENABLE_CUDA_64 "Enable CUDA (both 32 and 64 bit support, needs CUDA device capability of 2.0 or higher)" OFF)

option(USE_PYTHON2 "Use Python2" OFF)
option(USE_PYTHON3 "Use Python3" OFF)

if(ENABLE_CUDA_32 OR ENABLE_CUDA_64)
  set(ENABLE_CUDA ON)
endif(ENABLE_CUDA_32 OR ENABLE_CUDA_64)

##############################################################
###    Macros                                              ###
##############################################################

# Macro for adding src files from within subdirectories
macro(append_magneto_src DIR FILES)
  set(LIST)
  foreach(FILE ${FILES})
    list(APPEND LIST ${DIR}/${FILE})
  endforeach(FILE)
  set(MAGNETO_SRC ${MAGNETO_SRC};${LIST} PARENT_SCOPE)
endmacro(append_magneto_src)

# Macro for adding Cuda kernel src files from within subdirectories
macro(append_magneto_cusrc DIR FILES)
  set(LIST)
  foreach(FILE ${FILES})
    list(APPEND LIST ${DIR}/${FILE})
  endforeach(FILE)
  set(MAGNETO_CUSRC ${MAGNETO_CUSRC};${LIST} PARENT_SCOPE)
endmacro(append_magneto_cusrc)

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

# CMake version 2.8.8+ will select the highest Python library version including Python 3.x.
if(USE_PYTHON2)
  SET(Python_ADDITIONAL_VERSIONS 2.8 2.7 2.6)
elseif(USE_PYTHON3)
  SET(Python_ADDITIONAL_VERSIONS 3.3 3.2 3.1 3.0)
endif(USE_PYTHON2)
# CMake versions 2.8.7 and lower will always choose Python 2.x.

find_package(PythonLibs REQUIRED)

# OpenMP support (optional)
include(FindOpenMP)

##############################################################
###    CVode                                               ###
##############################################################

#add_executable(Cvode cvode/cvode.cpp ${MAGNETO_SRC})

#target_link_libraries(Cvode ${FFTW_LIBRARIES})

# - Try to find sundials_cvode
# Once done this will define
#  SUNDIALS_CVODE_FOUND - System has sundials_cvode
#  SUNDIALS_CVODE_INCLUDE_DIRS - The sundials_cvode include directories
#  SUNDIALS_CVODE_LIBRARIES - The libraries needed to use sundials_cvode
#  SUNDIALS_CVODE_DEFINITIONS - Compiler switches required for using sundials_cvode

find_package(PkgConfig)
pkg_check_modules(PC_SUNDIALS_CVODE QUIET sundials_cvode)
set(SUNDIALS_CVODE_DEFINITIONS ${PC_SUNDIALS_CVODE_CFLAGS_OTHER})

find_path(SUNDIALS_CVODE_INCLUDE_DIR cvode/cvode.h
  HINTS ${PC_SUNDIALS_CVODE_INCLUDEDIR} ${PC_SUNDIALS_CVODE_INCLUDE_DIRS}
  PATH_SUFFIXES sundials_cvode )

find_library(SUNDIALS_CVODE_LIBRARY NAMES cvode sundials_cvode
  HINTS ${PC_SUNDIALS_CVODE_LIBDIR} ${PC_SUNDIALS_CVODE_LIBRARY_DIRS} )

set(SUNDIALS_CVODE_LIBRARIES ${SUNDIALS_CVODE_LIBRARY} )
set(SUNDIALS_CVODE_INCLUDE_DIRS ${SUNDIALS_CVODE_INCLUDE_DIR} )

set(SUNDIALS_NVECSERIAL ${PC_SUNDIALS_NVECSERIAL_CFLAGS_OTHER})

find_path(SUNDIALS_NVECSERIAL_INCLUDE_DIR nvector/nvector_serial.h
  HINTS ${PC_SUNDIALS_NVECSERIAL_INCLUDEDIR} ${PC_SUNDIALS_NVECSERIAL_INCLUDE_DIRS}
  PATH_SUFFIXES sundials_nvecserial )

find_library(SUNDIALS_NVECSERIAL_LIBRARY NAMES nvector_serial sundials_nvecserial
  HINTS ${PC_SUNDIALS_NVECSERIAL_LIBDIR} ${PC_SUNDIALS_NVECSERIAL_LIBRARY_DIRS} )

set(SUNDIALS_NVECSERIAL_LIBRARIES ${SUNDIALS_NVECSERIAL_LIBRARY} )
set(SUNDIALS_NVECSERIAL_INCLUDE_DIRS ${SUNDIALS_NVECSERIAL_INCLUDE_DIR} )

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set SUNDIALS_CVODE_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(sundials_cvode  DEFAULT_MSG
  SUNDIALS_CVODE_LIBRARY SUNDIALS_CVODE_INCLUDE_DIR)

mark_as_advanced(SUNDIALS_CVODE_INCLUDE_DIR SUNDIALS_CVODE_LIBRARY )

##############################################################
###    Boost.python                                        ###
##############################################################

#FIND_PACKAGE(Boost 1.50.0)
#IF(Boost_FOUND)
#  INCLUDE_DIRECTORIES("${Boost_INCLUDE_DIRS}" "/usr/include/python2.7")
#  SET(Boost_USE_STATIC_LIBS OFF)
#  SET(Boost_USE_MULTITHREADED ON)
#  SET(Boost_USE_STATIC_RUNTIME OFF)
#  FIND_PACKAGE(Boost 1.50.0 COMPONENTS python)
#
#ELSEIF(NOT Boost_FOUND)
#  MESSAGE(FATAL_ERROR "Unable to find correct Boost version. Did you set BOOST_ROOT?")
#ENDIF()


##############################################################
###    Subdirectories                                      ###
##############################################################

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

include_directories(
  ${PYTHON_INCLUDE_PATH}
  ${PROJECT_SOURCE_DIR}/magneto
  ${PROJECT_SOURCE_DIR}/magneto/matrix
  ${PROJECT_BINARY_DIR}
)

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
swig_link_libraries(${SWIG_MODULE_NAME} ${FFTW_LIBRARIES} ${SUNDIALS_CVODE_LIBRARIES} ${SUNDIALS_NVECSERIAL_LIBRARIES})
#swig_link_libraries(${SWIG_MODULE_NAME} ${FFTW_LIBRARIES} ${SUNDIALS_CVODE_LIBRARIES} ${SUNDIALS_NVECSERIAL_LIBRARIES} ${Boost_LIBRARIES})

# Link Cuda Kernels to Swig module
if(ENABLE_CUDA)
  swig_link_libraries(${SWIG_MODULE_NAME} magneto_cu_parts)
endif(ENABLE_CUDA)

##############################################################
###    Don't forget to create config.h                     ###
##############################################################

# Add configuration file config.h
configure_file(
  "${PROJECT_SOURCE_DIR}/magneto/config.h.in"
  "${PROJECT_BINARY_DIR}/config.h"
)

##############################################################
###    Tests                                               ###
##############################################################

ENABLE_TESTING()
add_test(NAME allTests WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/.. COMMAND python${PYTHON_MAJOR} run-all-tests.py)

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
else(ENABLE_CUDA)
  # copy builded destinations of cpu symlinks
  install(FILES build/magneto_cpu.py build/_magneto_cpu.so DESTINATION "${PYTHON_SITE_PACKAGES}/magnum")
endif(ENABLE_CUDA)
