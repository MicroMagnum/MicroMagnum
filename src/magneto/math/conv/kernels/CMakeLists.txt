set(SRC
  cpu_copy_pad.cpp
  cpu_copy_unpad.cpp
  cpu_multiplication.cpp
  cpu_transpose_unpad.cpp
  cpu_transpose_zeropad.cpp

  Transposer_CPU.cpp
  Transformer_CPU.cpp
)

set(SRC_FOR_CUDA
  cuda_transpose_unpad.cpp
  cuda_transpose_zeropad.cpp

  Transformer_CUDA.cpp
  Transposer_CUDA.cpp
)

set(CUSRC
  cuda_copy_pad.cu
  cuda_copy_unpad.cu
  cuda_multiplication.cu
  cuda_transpose_unpad_2d.cu
  cuda_transpose_unpad_3d.cu
  cuda_transpose_zeropad_2d.cu
  cuda_transpose_zeropad_3d.cu
)

append_sources(${CMAKE_CURRENT_SOURCE_DIR} "${SRC}" "${SRC_FOR_CUDA}" "${CUSRC}")
