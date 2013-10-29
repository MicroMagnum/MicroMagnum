set(SRC
  runge_kutta.cpp
  runge_kutta_cpu.cpp
)

set(CUSRC
  runge_kutta_cuda.cu
)

append_sources(${CMAKE_CURRENT_SOURCE_DIR} "${SRC}" "" "${CUSRC}")
