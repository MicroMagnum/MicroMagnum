set(SRC
  runge_kutta.cpp
  runge_kutta_cpu.cpp
)

if(HAVE_CVODE)
  set(SRC ${SRC}
    cvode.cpp
    diffeq.cpp
    )
endif(HAVE_CVODE)

if(ENABLE_CUDA)
  set(SRC ${SRC}
    )
endif(ENABLE_CUDA)

# Add Cuda specific sources
set(CUSRC
  runge_kutta_cuda.cu
)

append_magneto_src(${CMAKE_CURRENT_SOURCE_DIR} "${SRC}")
append_magneto_cusrc(${CMAKE_CURRENT_SOURCE_DIR} "${CUSRC}")
