set(SRC
  gradient.cpp
  ScaledAbsMax.cpp
)

# Add Cuda specific sources
set(CUSRC
  gradient_cuda.cu
  ScaledAbsMax_cuda.cu
)

append_sources(${CMAKE_CURRENT_SOURCE_DIR} "${SRC}" "" "${CUSRC}")
