%{
#include "math/conv/SymmetricMatrixVectorConvolution_FFT.h"
#include "math/conv/SymmetricMatrixVectorConvolution_Simple.h"
#include "math/conv/AntisymmetricMatrixVectorConvolution_FFT.h"
#include "math/conv/VectorVectorConvolution_FFT.h"
%}

class SymmetricMatrixVectorConvolution_FFT
{
public:
	SymmetricMatrixVectorConvolution_FFT(const Matrix &lhs, int dim_x, int dim_y, int dim_z);
	virtual ~SymmetricMatrixVectorConvolution_FFT();
	virtual void execute(const VectorMatrix &rhs, VectorMatrix &res);
};

class SymmetricMatrixVectorConvolution_Simple
{
public:
	SymmetricMatrixVectorConvolution_Simple(const Matrix &lhs, int dim_x, int dim_y, int dim_z);
	virtual ~SymmetricMatrixVectorConvolution_Simple();
	virtual void execute(const VectorMatrix &rhs, VectorMatrix &res);
};

class AntisymmetricMatrixVectorConvolution_FFT
{
public:
	AntisymmetricMatrixVectorConvolution_FFT(const Matrix &lhs, int dim_x, int dim_y, int dim_z);
	virtual ~AntisymmetricMatrixVectorConvolution_FFT();
	virtual void execute(const VectorMatrix &rhs, VectorMatrix &res);
};

class VectorVectorConvolution_FFT
{
public:
	VectorVectorConvolution_FFT(const Matrix &lhs, int dim_x, int dim_y, int dim_z, double delta_x, double delta_y, double delta_z);
	virtual ~VectorVectorConvolution_FFT();

	virtual void execute(const VectorMatrix &rhs, VectorMatrix &res);
};

%{
#include "math/gradient.h"
#include "math/ScaledAbsMax.h"
%}

void gradient(double delta_x, double delta_y, double delta_z, const Matrix &pot, VectorMatrix &field);
double scaled_abs_max(VectorMatrix &M, Matrix &scale);

