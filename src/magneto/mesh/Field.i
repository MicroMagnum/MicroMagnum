%{
#include "mesh/Field.h"
%}

class Field : public Matrix
{
public:
        Field(const RectangularMesh &mesh);
        virtual ~Field();
        
        RectangularMesh getMesh() const;

        %extend {
        %pythoncode {

          def __repr__(self):
            return "Field(%r)" % self.mesh
          
          def interpolate(self, mesh):
            # %{ Get matrix with interpolated values in 'interp_mat' %}
            need_interpolate = (self.mesh.num_nodes != mesh.num_nodes)
            if need_interpolate:
              nx, ny, nz = mesh.num_nodes # new size (in number of cells)
              interp_mat = magneto.linearInterpolate(self, magneto.Shape(nx, ny, nz))
            else:
              interp_mat = self # no need to interpolate..
          
            # %{ Create interpolated vector field from matrix 'interp_mat' %}
            result = Field(mesh)
            result.assign(interp_mat)
            return result
        }
        }
};

%pythoncode {

Field.mesh = property(Field.getMesh)

}
