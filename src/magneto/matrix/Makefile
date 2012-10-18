ENABLE_CUDA = no
EXE = main

SRC += main.cpp

SRC += device/Array.cpp
SRC += device/Device.cpp
SRC += device/DeviceManager.cpp

SRC += device/cpu/CPUArray.cpp
SRC += device/cpu/CPUDevice.cpp

SRC += matrix/AbstractMatrix.cpp
SRC += matrix/scalar/Matrix.cpp
SRC += matrix/scalar/Matrix_accessor.cpp
SRC += matrix/vector/Vector3d.cpp
SRC += matrix/vector/VectorMatrix.cpp
SRC += matrix/vector/VectorMatrix_accessor.cpp
SRC += matrix/complex/ComplexMatrix.cpp

SRC += Shape.cpp
SRC += matty.cpp

####################################
# II. Include additional libraries #
####################################

# Common
LDFLAGS += -lm
INCLUDE_FLAGS += -I. `python-config --includes`

# FFTW
LDFLAGS += -lfftw3 

#####################################
# III.                              #
#####################################

CC = g++ -fpic -g
DEFINES = 

######################################
# CUDA support                       #
######################################

ifeq ($(ENABLE_CUDA),yes)

CUDA_SRC=
CUDA_OBJ += ${CUDA_SRC:.cu=.o}

DEFINES += -DHAVE_CUDA
INCLUDE_FLAGS += -I${CUDA_HOME}/include
LDFLAGS += -lcudart -L${CUDA_HOME}/lib64
LDFLAGS += -lcufft -lcublas

%.o: %.cu
	nvcc -Xcompiler -fpic ${DEFINES} -O3 --use_fast_math ${INCLUDE_FLAGS} -c $< -o $@

endif

#####################################
# Make rules                        #
#####################################

OBJ = ${SRC:.cpp=.o}

%.o: %.cpp
	${CC} ${DEFINES} ${CPPFLAGS} ${INCLUDE_FLAGS} -c $< -o $@


all: ${OBJ} ${CUDA_OBJ}
	$(CC) ${OBJ} ${CUDA_OBJ} ${LDFLAGS} -o ${EXE} 

clean:
	-rm -f ${OBJ} ${CUDA_OBJ} ${EXE}
	-rm -f matty.py matty_wrap.cpp _matty.so


matty_wrap.cpp: matty.i
	swig -O -c++ -python -o matty_wrap.cpp matty.i

_matty.so: matty_wrap.o ${OBJ} ${CUDA_OBJ}
	$(CC) matty_wrap.o ${OBJ} ${CUDA_OBJ} -shared -o _matty.so

matty.py: _matty.so
