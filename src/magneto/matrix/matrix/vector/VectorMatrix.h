#ifndef VECTORMATRIX_H
#define VECTORMATRIX_H

#include "config.h"

#include "matrix/AbstractMatrix.h"
#include "Vector3d.h"

#include "VectorMatrix_accessor.h"
#ifdef HAVE_CUDA
#include "VectorMatrix_cuda_accessor.h"
#endif

namespace matty {

class VectorMatrix : public AbstractMatrix
{
public:
	typedef      VectorMatrixAccessor accessor;
	typedef ConstVectorMatrixAccessor const_accessor;
#ifdef HAVE_CUDA
	typedef      VectorMatrixCU32Accessor cuda_accessor;
	typedef ConstVectorMatrixCU32Accessor const_cuda_accessor;
	typedef      VectorMatrixCU32Accessor cu32_accessor;
	typedef ConstVectorMatrixCU32Accessor const_cu32_accessor;
#ifdef HAVE_CUDA_64
	typedef      VectorMatrixCU64Accessor cu64_accessor;
	typedef ConstVectorMatrixCU64Accessor const_cu64_accessor;
#endif
#endif

	VectorMatrix();
	VectorMatrix(const Shape &shape);
	VectorMatrix(const VectorMatrix &other);
	VectorMatrix &operator=(VectorMatrix other);
	virtual ~VectorMatrix();

	void clear();
	void fill(Vector3d value);
	void assign(const VectorMatrix &other);
	void scale(double factor);
	void scale(const Vector3d &factors);
	void add(const VectorMatrix &op, double scale = 1.0);
	void randomize();

	void normalize(double len);
	void normalize(const class Matrix &len);

	//Vector3d mininum() const;
	Vector3d maximum() const;
	Vector3d average() const;
	Vector3d sum() const;

	double absMax() const;
	double dotSum(const VectorMatrix &other) const;

	Vector3d getUniformValue() const;

	Array *getArray(int dev, int comp) const;
};

std::ostream &operator<<(std::ostream &out, const VectorMatrix &mat);

} // ns

#include "VectorMatrix_accessor.h"
#ifdef HAVE_CUDA
#include "VectorMatrix_cuda_accessor.h"
#endif

#endif

