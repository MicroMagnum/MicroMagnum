%{
#include "mesh/VectorField.h"
%}

class VectorField : public VectorMatrix
{
public:
        VectorField(const RectangularMesh &mesh);
        virtual ~VectorField();
        
        RectangularMesh getMesh() const;
};

%pythoncode {

def VectorField__repr__(self):
  return "VectorField(%r)" % self.mesh

def VectorField_initFromFunction(self, init_fn):
  mesh = self.mesh
  for idx in range(mesh.total_nodes):
    self.set(idx, init_fn(mesh, mesh.getPosition(idx)))

def VectorField_findExtremum(self, z_slice=0, component=0): # TODO: Better name (only the xy-Plane is searched)
  cell = magneto.findExtremum(self, z_slice, component)
  return (
    (0.5 + cell[0]) * self.mesh.delta[0],
    (0.5 + cell[1]) * self.mesh.delta[1],
    (0.5 + cell[2]) * self.mesh.delta[2]
  )

def VectorField_interpolate(self, mesh):
  # Get matrix with interpolated values in 'interp_mat'   
  need_interpolate = (self.mesh.num_nodes != mesh.num_nodes)
  if need_interpolate:
    nx, ny, nz = mesh.num_nodes # new size (in number of cells)
    interp_mat = magneto.linearInterpolate(self, magneto.Shape(nx, ny, nz))
  else:
    interp_mat = self # no need to interpolate..

  # Create interpolated vector field from matrix 'interp_mat'
  result = VectorField(mesh)
  result.assign(interp_mat)
  return result

VectorField.mesh = property(VectorField.getMesh)
VectorField.initFromFunction = VectorField_initFromFunction
VectorField.findExtremum = VectorField_findExtremum
VectorField.interpolate = VectorField_interpolate
VectorField.__repr__ = VectorField__repr__

}
