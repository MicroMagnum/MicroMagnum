set(SRC
  # anisotropy
  anisotropy/anisotropy.cpp
  anisotropy/anisotropy_cpu.cpp

  # demag
  demag/phi/demag_phi_tensor.cpp
  demag/tensor_round.cpp
  demag/demag_static.cpp
  demag/tensor.cpp
  demag/old/demag_old.cpp
  demag/demag_tensor.cpp

  # exchange
  exchange/fdm_exchange.cpp
  exchange/fdm_exchange_cpu.cpp

  # io
  io/OMFHeader.cpp
  io/OMFExport.cpp
  io/OMFImport.cpp

  # llge
  llge/llge.cpp
  llge/llge_cpu.cpp

  # spintorque
  spintorque/fdm_zhangli.cpp
  spintorque/fdm_zhangli_cpu.cpp
  spintorque/fdm_slonchewski.cpp
)

set(CUSRC
  anisotropy/anisotropy_cuda.cu
  exchange/fdm_exchange_cuda.cu
  llge/llge_cuda.cu
  spintorque/fdm_zhangli_cuda.cu
)

append_magneto_src(${CMAKE_CURRENT_SOURCE_DIR} "${SRC}")
append_magneto_cusrc(${CMAKE_CURRENT_SOURCE_DIR} "${CUSRC}")
