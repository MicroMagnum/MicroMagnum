#ifndef FIELD_H
#define FIELD_H

#include "matrix/matty.h"
#include "RectangularMesh.h"

class Field : public Matrix
{
public:
	Field(const RectangularMesh &mesh);
	virtual ~Field();

	const RectangularMesh &getMesh() const { return mesh; }

private:
	const RectangularMesh mesh;
};

#endif
