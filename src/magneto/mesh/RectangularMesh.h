#ifndef RECTANGULAR_MESH_H
#define RECTANGULAR_MESH_H

#include <string>

class RectangularMesh
{
public:
	RectangularMesh(int nx, int ny, int nz, double dx, double dy, double dz, std::string pbc, int pbc_reps)
		: nx(nx), ny(ny), nz(nz), dx(dx), dy(dy), dz(dz), pbc(pbc), pbc_reps(pbc_reps)
	{
	}

	bool isCompatible(const RectangularMesh &other) const
	{
		// XXX: What about periodic boundary conditions?
		return    nx == other.nx && ny == other.ny && nz == other.nz
		       && dx == other.dx && dy == other.dy && dz == other.dz;
	}

	double getCellVolume() const { return dx * dy * dz; }
	double getTotalNodes() const { return nx * ny * nz; }
	double getVolume() const { return getCellVolume() * getTotalNodes(); }

        void getNumNodes(int &nx, int &ny, int &nz) const { nx = this->nx; ny = this->ny; nz = this->nz; }
        void getDelta(int &dx, int &dy, int &dz) const { dx = this->dx; dy = this->dy; dz = this->dz; }
        void getSize(double &size_x, double &size_y, double &size_z) const { size_x = nx*dx; size_y = ny*dy; size_z = nz*dz; }

	void getPeriodicBC(std::string &pbc, int &pbc_reps) const { pbc = this->pbc; pbc_reps = this->pbc_reps; }

	void getPosition(int linidx, double &pos_x, double &pos_y, double &pos_z) // Returns the middle point of the cell with linear index 'linidx'.
	{
		const int stride_x = 1;
		const int stride_y = nx * stride_x;
		const int stride_z = ny * stride_y;

		const int z = (linidx           ) / stride_z;
		const int y = (linidx % stride_z) / stride_y;
		const int x = (linidx % stride_y) / stride_x;

		// return pos (of cell center, hence +0.5)
		pos_x = (x+0.5)*dx;
		pos_y = (y+0.5)*dy;
		pos_z = (z+0.5)*dz;
	}

public:
	const int nx, ny, nz;
	const double dx, dy, dz;
	std::string pbc;
	int pbc_reps;
};

#endif
