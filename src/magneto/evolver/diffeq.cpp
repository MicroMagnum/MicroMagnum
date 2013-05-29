/*
 * Copyright 2012 by the Micromagnum authors.
 *
 * This file is part of MicroMagnum.
 * 
 * MicroMagnum is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * MicroMagnum is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with MicroMagnum.  If not, see <http://www.gnu.org/licenses/>.
 */

//#include "Magneto.h"
#include "diffeq.h"
//#include <stdlib.h>
#include <iostream>
#include <cvode/cvode.h>

using namespace matty;

DiffEq::DiffEq(VectorMatrix &My)
  : _My(My),_Mydot(My)
{
}

DiffEq::~DiffEq()
{
}

void DiffEq::diffN(const N_Vector& Ny, N_Vector& Nydot, realtype t)
{
  /*
   * convert Ny->My
   * diff()
   * convert Mydot->Nydot
   */

  getVectorMatrix(Ny,_My);

  //_Mydot = diff(_My);
  diffX(_My,_Mydot,t);

  getN_Vector(_Mydot,Nydot);
}


/**
 * nvec muss mit der richtigen Größe initialisiert sein (N_VNew_Serial(3*size)).
 */
void DiffEq::getN_Vector(const VectorMatrix& mat, N_Vector& nvec)
{
  int dim_x = mat.dimX();
  int dim_y = mat.dimY();
  int dim_z = mat.dimZ();
	const int dim_xy = dim_x * dim_y;

  VectorMatrix::const_accessor Macc(mat);

	for (int z=0; z<dim_z; ++z)
  for (int y=0; y<dim_y; ++y)
  for (int x=0; x<dim_x; ++x) 
  {
    const int i = z*dim_xy + y*dim_x + x; // linear index of (x,y,z)
    Vector3d vec3 = Macc.get(i);
    Ith(nvec,3*i+0) = vec3[0];
    Ith(nvec,3*i+1) = vec3[1];
    Ith(nvec,3*i+2) = vec3[2];
  }
}

void DiffEq::getVectorMatrix(const N_Vector& vec, VectorMatrix& mat)
{
  int dim_x = mat.dimX();
  int dim_y = mat.dimY();
  int dim_z = mat.dimZ();
	const int dim_xy = dim_x * dim_y;

  VectorMatrix::accessor Macc(mat);

	for (int z=0; z<dim_z; ++z)
  for (int y=0; y<dim_y; ++y)
  for (int x=0; x<dim_x; ++x) {
    const int i = z*dim_xy + y*dim_x + x; // linear index of (x,y,z)

    Vector3d vec3;

    vec3[0] = Ith(vec,3*i+0);
    vec3[1] = Ith(vec,3*i+1);
    vec3[2] = Ith(vec,3*i+2);
    Macc.set(i, vec3);
  }
}

void DiffEq::printN_Vector(const N_Vector& nvec)
{
  for (int i=0; i<size()/3; i++)
  {
    std::cout << "(" << Ith(nvec,3*i+0) << ","
                     << Ith(nvec,3*i+1) << ","
                     << Ith(nvec,3*i+2) << ")\n";
  }
}

void DiffEq::printVectorMatrix(const VectorMatrix& mat)
{
  int dim_x = mat.dimX();
  int dim_y = mat.dimY();
  int dim_z = mat.dimZ();
  const int dim_xy = dim_x * dim_y;

  std::cout << "VectorMatrix size: " << mat.size() << "\n";
  std::cout << "VectorMatrix dimX: " << mat.dimX() << "\n";
  std::cout << "VectorMatrix dimY: " << mat.dimY() << "\n";
  std::cout << "VectorMatrix dimZ: " << mat.dimZ() << "\n";

  {
    VectorMatrix::const_accessor Macc(mat);

    for (int z=0; z<dim_z; ++z) {
      for (int y=0; y<dim_y; ++y) {	
        for (int x=0; x<dim_x; ++x) {
          const int i = z*dim_xy + y*dim_x + x; // linear index of (x,y,z)
          std::cout << Macc.get(i);
          std::cout << std::endl;
        }
      }
    }
  }
}

void DiffEq::printOutput(const realtype &t, const N_Vector& nvec)
{
  std::cout << "Time = " << t << std::endl;
  printN_Vector(nvec);
}

void DiffEq::matrixTest(VectorMatrix mat)
{
  int dim_x = mat.dimX();
  int dim_y = mat.dimY();
  int dim_z = mat.dimZ();
  const int dim_xy = dim_x * dim_y;

  std::cout << "matrixTest size: " << mat.size() << "\n";
  std::cout << "matrixTest dimX: " << mat.dimX() << "\n";
  std::cout << "matrixTest dimY: " << mat.dimY() << "\n";
  std::cout << "matrixTest dimZ: " << mat.dimZ() << "\n";

  {
    VectorMatrix::const_accessor Macc(mat);

    for (int z=0; z<dim_z; ++z) {
      for (int y=0; y<dim_y; ++y) {	
        for (int x=0; x<dim_x; ++x) {
          const int i = z*dim_xy + y*dim_x + x; // linear index of (x,y,z)
          std::cout << Macc.get(i);
          std::cout << std::endl;
        }
      }
    }
  }

  /*
   * Konverter Test
   */

  N_Vector x = N_VNew_Serial(3*mat.size());
  getN_Vector(mat,x);
  realtype x1,x2,x3;

  std::cout << std::endl;
  std::cout << "konvertiert:" << std::endl;
  for (int i=0; i<mat.size(); ++i) {
    Ith(x,3*i+1) += RCONST(1); 
    Ith(x,3*i+2) += RCONST(2); 
    Ith(x,3*i+3) += RCONST(3);

    x1 = Ith(x,3*i+1);
    x2 = Ith(x,3*i+2);
    x3 = Ith(x,3*i+3);

    std::cout <<"(" << x1 << "," << x2 << "," << x3 << ")";
    std::cout << std::endl;
  }

  /*
   * Zurück konvertieren
   */
  {
    getVectorMatrix(x, mat);
    VectorMatrix::const_accessor Macc(mat);

    std::cout << "\nzurück konvertiert:\n";
    for (int z=0; z<dim_z; ++z) {
      for (int y=0; y<dim_y; ++y) {	
        for (int x=0; x<dim_x; ++x) {
          const int i = z*dim_xy + y*dim_x + x; // linear index of (x,y,z)
          std::cout << Macc.get(i);
          std::cout << std::endl;
        }
      }
    }
  }
}

int DiffEq::size()
{
  return 3*_My.size();
}


void DiffEq::saveStateC(N_Vector yout)
{
  getVectorMatrix(yout, _My);
  saveState(_My);
}





/*
 * C++ dummy. Should not be called!
 */
void DiffEq::diffX(const VectorMatrix &My, VectorMatrix &Mydot, double t)
{
  fprintf(stderr,"diffX: Python Director does not work!\n");
  exit(0);
}

/*
 * C++ dummy. Should not be called!
 */
VectorMatrix DiffEq::getY()
{
  VectorMatrix y;

  fprintf(stderr,"getY: Python Director does not work!\n");
  exit(0);

  return y;
}

/*
 * C++ dummy. Should not be called!
 */
VectorMatrix DiffEq::diff(const VectorMatrix &My)
{
  fprintf(stderr,"diff: Python Director does not work!\n");
  exit(0);

  return My;
}

/*
 * C++ dummy. Should not be called!
 */
void DiffEq::saveState(VectorMatrix mat)
{
  fprintf(stderr,"saveState: Python Director does not work!\n");
  exit(0);
}

/*
 * C++ dummy. Should not be called!
 */
void DiffEq::saveTime(double t)
{
  fprintf(stderr,"saveTime: Python Director does not work!\n");
  exit(0);
}

/*
 * C++ dummy. Should not be called!
 */
void DiffEq::substep()
{
  fprintf(stderr,"substep: Python Director does not work!\n");
  exit(0);
}
