set(SRC
  gradient.cpp
  ScaledAbsMax.cpp
)

# Add Cuda specific sources
set(CUSRC
  gradient_cuda.cu
  ScaledAbsMax_cuda.cu
)

append_magneto_src(${CMAKE_CURRENT_SOURCE_DIR} "${SRC}")
append_magneto_cusrc(${CMAKE_CURRENT_SOURCE_DIR} "${CUSRC}")
