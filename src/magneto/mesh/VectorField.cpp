#include "VectorField.h"

static Shape mesh_to_shape(const RectangularMesh &mesh)
{
	int nx, ny, nz; 
	mesh.getNumNodes(nx, ny, nz);
	return Shape(nx, ny, nz);
}

VectorField::VectorField(const RectangularMesh &mesh)
	: VectorMatrix(mesh_to_shape(mesh)), mesh(mesh)
{
}

VectorField::~VectorField()
{
}
