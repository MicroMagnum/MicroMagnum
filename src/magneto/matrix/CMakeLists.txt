set(SRC
  Shape.cpp
  matty.cpp

  device/Array.cpp
  device/Device.cpp
  device/DeviceManager.cpp
  
  device/cpu/CPUArray.cpp
  device/cpu/CPUDevice.cpp
  
  matrix/AbstractMatrix.cpp
  matrix/scalar/Matrix.cpp
  matrix/scalar/Matrix_accessor.cpp
  matrix/vector/Vector3d.cpp
  matrix/vector/VectorMatrix.cpp
  matrix/vector/VectorMatrix_accessor.cpp
  matrix/complex/ComplexMatrix.cpp

  ext/LinearInterpolate.cpp
  ext/Extremum.cpp
  ext/FFT.cpp
)

# Add Cuda specific sources
if(ENABLE_CUDA)
  set(SRC ${SRC}
    device/cuda/CUDADevice.cpp
    device/cuda/CUDAArray.cpp
    device/cuda_tools.cpp
    matrix/scalar/Matrix_cuda_accessor.cpp
    matrix/vector/VectorMatrix_cuda_accessor.cpp
  )
endif(ENABLE_CUDA)

set(CUSRC
  device/cuda/kernels_reduce.cu
  device/cuda/kernels_simple.cu
)

append_magneto_src(${CMAKE_CURRENT_SOURCE_DIR} "${SRC}")
append_magneto_cusrc(${CMAKE_CURRENT_SOURCE_DIR} "${CUSRC}")
