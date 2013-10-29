set(SRC
  MatrixVectorConvolution_FFT.cpp
  SymmetricMatrixVectorConvolution_Simple.cpp
  SymmetricMatrixVectorConvolution_FFT.cpp
  AntisymmetricMatrixVectorConvolution_FFT.cpp

  TensorFieldSetup.cpp

  VectorVectorConvolution_FFT.cpp
)

append_sources(${CMAKE_CURRENT_SOURCE_DIR} "${SRC}" "" "")
