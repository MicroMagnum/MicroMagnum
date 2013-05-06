#ifndef VECTOR_FIELD_H
#define VECTOR_FIELD_H

#include "matrix/matty.h"
#include "RectangularMesh.h"

class VectorField : public VectorMatrix
{
public:
	VectorField(const RectangularMesh &mesh);
	virtual ~VectorField();

	const RectangularMesh &getMesh() const { return mesh; }

private:
	const RectangularMesh mesh;
};

#endif
