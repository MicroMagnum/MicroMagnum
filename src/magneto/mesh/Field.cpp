#include "Field.h"

static Shape mesh_to_shape(const RectangularMesh &mesh)
{
	int nx, ny, nz; 
	mesh.getNumNodes(nx, ny, nz);
	return Shape(nx, ny, nz);
}

Field::Field(const RectangularMesh &mesh)
	: Matrix(mesh_to_shape(mesh)), mesh(mesh)
{
}

Field::~Field()
{
}
