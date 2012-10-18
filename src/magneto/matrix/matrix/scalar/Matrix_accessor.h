#ifndef MATRIX_ACCESSOR_H
#define MATRIX_ACCESSOR_H

#include "config.h"
#include "restrict.h"
#include "matrix/Shape.h"

namespace matty {

class Matrix;

#define MATRIX_ACCESSOR_DEFS(qualifier)                                                                                          \
public:                                                                                                                         \
	qualifier double &at(int x0) { return data[x0]; }                                                                       \
	qualifier double &at(int x0, int x1) { return at(x0 + x1*strides[1]); }                                                 \
	qualifier double &at(int x0, int x1, int x2) { return at(x0 + x1*strides[1] + x2*strides[2]); }                         \
	qualifier double &at(int x0, int x1, int x2, int x3) { return at(x0 + x1*strides[1] + x2*strides[2] + x3*strides[3]); } \
	qualifier double *ptr() { return data; }                                                                                \
private:                                                                                                                        \
	qualifier Matrix &mat;                                                                                                  \
	qualifier double * RESTRICT data;                                                                                        \
	int strides[4];

class MatrixAccessor_read_only // read-only
{
public:
	MatrixAccessor_read_only(const Matrix &mat);
	~MatrixAccessor_read_only();

	MATRIX_ACCESSOR_DEFS(const)
};

class MatrixAccessor_write_only // write-only
{
public:
	MatrixAccessor_write_only(Matrix &mat);
	~MatrixAccessor_write_only();

	MATRIX_ACCESSOR_DEFS()
};

class MatrixAccessor_read_write // read-write
{
public:
	MatrixAccessor_read_write(Matrix &mat);
	~MatrixAccessor_read_write();

	MATRIX_ACCESSOR_DEFS()
};

#undef MATRIX_ACCESSOR_DEFS

} // ns

#endif
