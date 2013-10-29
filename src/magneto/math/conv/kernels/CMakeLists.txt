set(SRC
  cpu_copy_pad.cpp
  cpu_copy_unpad.cpp
  cpu_multiplication.cpp
  cpu_transpose_unpad.cpp
  cpu_transpose_zeropad.cpp

  Transposer_CPU.cpp
  Transformer_CPU.cpp
)

if(ENABLE_CUDA)
  set(SRC ${SRC}
    cuda_transpose_unpad.cpp
    cuda_transpose_zeropad.cpp

    Transformer_CUDA.cpp
    Transposer_CUDA.cpp
  )
endif(ENABLE_CUDA)

# Add Cuda specific sources
set(CUSRC
  cuda_copy_pad.cu
  cuda_copy_unpad.cu
  cuda_multiplication.cu
  cuda_transpose_unpad_2d.cu
  cuda_transpose_unpad_3d.cu
  cuda_transpose_zeropad_2d.cu
  cuda_transpose_zeropad_3d.cu
)

append_magneto_src(${CMAKE_CURRENT_SOURCE_DIR} "${SRC}")
append_magneto_cusrc(${CMAKE_CURRENT_SOURCE_DIR} "${CUSRC}")
