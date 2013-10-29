set(SRC
  MatrixVectorConvolution_FFT.cpp
  SymmetricMatrixVectorConvolution_Simple.cpp
  SymmetricMatrixVectorConvolution_FFT.cpp
  AntisymmetricMatrixVectorConvolution_FFT.cpp

  TensorFieldSetup.cpp

  VectorVectorConvolution_FFT.cpp
)

append_magneto_src(${CMAKE_CURRENT_SOURCE_DIR} "${SRC}")
append_magneto_cusrc(${CMAKE_CURRENT_SOURCE_DIR} "${CUSRC}")
