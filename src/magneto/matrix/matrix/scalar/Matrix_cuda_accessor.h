#ifndef MATRIX_CUDA_ACCESSOR_H
#define MATRIX_CUDA_ACCESSOR_H

#include "config.h"

namespace matty {

class Matrix;

class MatrixCU32Accessor
{
public:
	MatrixCU32Accessor(Matrix &mat);
	~MatrixCU32Accessor();

	float *ptr() { return data; }

private:
	Matrix &mat;
	float *data;
};

class ConstMatrixCU32Accessor
{
public:
	ConstMatrixCU32Accessor(const Matrix &mat);
	~ConstMatrixCU32Accessor();

	const float *ptr() { return data; }

private:
	const Matrix &mat;
	const float *data;
};

#ifdef HAVE_CUDA_64
class MatrixCU64Accessor
{
public:
	MatrixCU64Accessor(Matrix &mat);
	~MatrixCU64Accessor();

	double *ptr() { return data; }

private:
	Matrix &mat;
	double *data;
};

class ConstMatrixCU64Accessor
{
public:
	ConstMatrixCU64Accessor(const Matrix &mat);
	~ConstMatrixCU64Accessor();

	const double *ptr() { return data; }

private:
	const Matrix &mat;
	const double *data;
};
#endif

// Structs to help with writing generic code.
template <typename real>
struct Matrix_cuda_accessor {};

template <typename real>
struct Matrix_const_cuda_accessor {};

template <>
struct Matrix_cuda_accessor<float> {
	typedef MatrixCU32Accessor t;
};

template <>
struct Matrix_const_cuda_accessor<float> {
	typedef ConstMatrixCU32Accessor t;
};
#ifdef HAVE_CUDA_64
template <>
struct Matrix_cuda_accessor<double> {
	typedef MatrixCU64Accessor t;
};

template <>
struct Matrix_const_cuda_accessor<double> {
	typedef ConstMatrixCU64Accessor t;
};
#endif

} // ns

#endif
