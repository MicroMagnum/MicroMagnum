%{
#include "mesh/VectorField.h"
%}

class VectorField : public VectorMatrix
{
public:
        VectorField(const RectangularMesh &mesh);
        virtual ~VectorField();
        
        RectangularMesh getMesh() const;

        %extend {
        %pythoncode {

          def __repr__(self):
            return "VectorField(%r)" % self.mesh
          
          def initFromFunction(self, init_fn):
            mesh = self.mesh
            for idx in range(mesh.total_nodes):
              self.set(idx, init_fn(mesh, mesh.getPosition(idx)))
          
          def findExtremum(self, z_slice=0, component=0): # TODO: Better name (only the xy-Plane is searched)
            import magnum.magneto as magneto
            cell = magneto.findExtremum(self, z_slice, component)
            return (
              (0.5 + cell[0]) * self.mesh.delta[0],
              (0.5 + cell[1]) * self.mesh.delta[1],
              (0.5 + cell[2]) * self.mesh.delta[2]
            )
          
          def interpolate(self, mesh):
            # %{ get matrix with interpolated values in 'interp_mat' %}
            import magnum.magneto as magneto
            need_interpolate = (self.mesh.num_nodes != mesh.num_nodes)
            if need_interpolate:
              nx, ny, nz = mesh.num_nodes # new size (in number of cells)
              interp_mat = magneto.linearInterpolate(self, magneto.Shape(nx, ny, nz))
            else:
              interp_mat = self # no need to interpolate..
          
            # %{ Create interpolated vector field from matrix 'interp_mat' %}
            result = VectorField(mesh)
            result.assign(interp_mat)
            return result

        }
        }
};

%pythoncode {

VectorField.mesh = property(VectorField.getMesh)

}
