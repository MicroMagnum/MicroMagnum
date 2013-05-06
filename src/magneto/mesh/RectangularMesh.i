%{
#include "mesh/RectangularMesh.h"
%}

class RectangularMesh
{
public:
        RectangularMesh(int nx, int ny, int nz, double dx, double dy, double dz, const std::string &pbc = "", int pbc_reps = -1);

        bool isCompatible(const RectangularMesh &other) const;
        
        double getCellVolume() const;
        int getTotalNodes() const;
        double getVolume() const;

        void getNumNodes(int &OUTPUT, int &OUTPUT, int &OUTPUT);
        void getDelta(double &OUTPUT, double &OUTPUT, double &OUTPUT) const;
        void getSize(double &OUTPUT, double &OUTPUT, double &OUTPUT) const;
        void getPeriodicBC(std::string &OUTPUT, int &OUTPUT) const;
        void getPosition(int linidx, double &OUTPUT, double &OUTPUT, double &OUTPUT);

        %extend {

        %pythoncode {
          def __repr__(self):
            return "RectangularMesh(%r, %r, periodic_bc=%r, periodic_repeat=%r)" % (self.num_nodes, self.delta, self.periodic_bc[0], self.periodic_bc[1])
          
          def iterateCellIndices(self):
            """
            Returns iterator that iterates through all cell indices (x,y,z).
            Example:
              for x,y,z in mesh.iterateCellIndices():
                print(x,y,z)
            """
            import itertools
            return itertools.product(*map(range, self.num_nodes))
        }
        }
};

%pythoncode {

RectangularMesh.volume      = property(                   RectangularMesh.getVolume)
RectangularMesh.cell_volume = property(                   RectangularMesh.getCellVolume)
RectangularMesh.total_nodes = property(                   RectangularMesh.getTotalNodes)
RectangularMesh.num_nodes   = property(lambda self: tuple(RectangularMesh.getNumNodes(self)))
RectangularMesh.delta       = property(lambda self: tuple(RectangularMesh.getDelta(self)))
RectangularMesh.size        = property(lambda self: tuple(RectangularMesh.getSize(self)))
RectangularMesh.periodic_bc = property(lambda self: tuple(RectangularMesh.getPeriodicBC(self)))

}
