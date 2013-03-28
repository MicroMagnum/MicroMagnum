%{
#include "mesh/RectangularMesh.h"
%}

class RectangularMesh
{
public:
        RectangularMesh(int nx, int ny, int nz, double dx, double dy, double dz, std::string pbc, int pbc_reps);
        
        bool isCompatible(const RectangularMesh &other) const;
        
        double getCellVolume() const;
        double getTotalNodes() const;
        double getVolume() const;

        void getNumNodes(int &OUTPUT, int &OUTPUT, int &OUTPUT);
        void getDelta(int &OUTPUT, int &OUTPUT, int &OUTPUT) const;
        void getSize(double &OUTPUT, double &OUTPUT, double &OUTPUT) const;
        void getPeriodicBC(std::string &OUTPUT, int &OUTPUT) const;
        void getPosition(int linidx, double &OUTPUT, double &OUTPUT, double &OUTPUT);
};

