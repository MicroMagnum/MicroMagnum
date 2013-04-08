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

// System
#include <stdlib.h>
#include <iostream>
//#include <stdio.h>
//#include <math.h>

// CVode includes
#include "cvode.h"
#include <cvode/cvode.h>
#include <cvode/cvode_dense.h>       /* prototype for CVDense */
#include <nvector/nvector_serial.h>
//#include <cvode/cvode_band.h>
//#include <cvode/cvode_band.h>        /* use CVBAND linear solver */
//#include <cvode/cvode_diag.h>        /* use CVDIAG linear solver */
//#include <sundials/sundials_types.h>
//#include <sundials/sundials_math.h>
//#include <sundials/sundials_band.h>

// Magnum
//#include "config.h"
#include "Magneto.h"
#include "diffeq.h"
//#include "matrix/matty.h"
//#include "Vector3d.h"

Cvode::Cvode(VectorMatrix &My, VectorMatrix &Mydot, DiffEq &diff)
  : _My(My), _Mydot(Mydot), _Ny(), _Nydot(), _abstol(), _size(3*My.size()), _diff(diff)
{
  std::cout << "size: " << _size << "\n";

  _Ny = N_VNew_Serial(_size);
  _Nydot = N_VNew_Serial(_size);
  _abstol = N_VNew_Serial(_size);

  //getN_Vector(_My, _Ny);
  //getN_Vector(_Mydot, _Nydot);

  _reltol = 0.1;

  std::cout << "size: " << _size << std::endl;
  for (int i=1; i<=_size; ++i)
  {
    Ith(_abstol,i) = 0.1;
  }
}

Cvode::~Cvode()
{
}

static void PrintOutput(realtype t, N_Vector x, int size)
{
  std::cout << "\nErgebnis:\n";
  realtype x1,x2,x3;
  for (int i=0; i<size; ++i) {
    Ith(x,3*i+1) += RCONST(1); 
    Ith(x,3*i+2) += RCONST(2); 
    Ith(x,3*i+3) += RCONST(3);

    x1 = Ith(x,3*i+1);
    x2 = Ith(x,3*i+2);
    x3 = Ith(x,3*i+3);

    std::cout <<"(" << x1 << "," << x2 << "," << x3 << ")";
    std::cout << std::endl;
  }
}

void Cvode::matrixTest(VectorMatrix mat)
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

void Cvode::one(int i) 
{
  std::cout << "one from c++\n";
}

int Cvode::cvodeTest() 
{
  one(123456);
  realtype t;
  N_Vector yout;
  void *cvode_mem;
  int flag, ans;
  VectorMatrix My, Mydot;
  //DiffEq user_data;
  //user_data.diff(My,Mydot);
  _diff.getY();
  _diff.diff(_My,_Mydot);

  yout = NULL;
  cvode_mem = NULL;
  //user_data = NULL;

  //y = N_VNew_Serial(3);
  //ydot = N_VNew_Serial(3);
  yout = N_VNew_Serial(_size);

  /* Call CVodeCreate to create the solver memory and specify the 
   * Backward Differentiation Formula and the use of a Newton iteration */
  cvode_mem = CVodeCreate(CV_BDF, CV_NEWTON);
  if (check_flag((void *)cvode_mem, (char *) "CVodeCreate", 0)) return(1);

  /* Set the pointer to user-defined data */
  flag = CVodeSetUserData(cvode_mem, &_diff);
  if(check_flag(&flag, (char *) "CVodeSetUserData", 1)) return(1);

  /* Call CVodeInit to initialize the integrator memory and specify the
   * user's right hand side function in y'=f(t,y), the inital time T0, and
   * the initial dependent variable vector y. */
  flag = CVodeInit(cvode_mem, callf, T0, _Ny);
  if (check_flag(&flag, (char *) "CVodeInit", 1)) return(1);

  /* Call CVodeSVtolerances to specify the scalar relative tolerance
   * and vector absolute tolerances */
  flag = CVodeSVtolerances(cvode_mem, _reltol, _abstol);
  if (check_flag(&flag, (char *) "CVodeSVtolerances", 1)) return(1);

  /* Call CVDense to specify the CVDENSE dense linear solver */
  flag = CVDense(cvode_mem, 3);
  if (check_flag(&flag, (char *) "CVDense", 1)) return(1);

  std::cout << "cvode 1\n";

  flag = CVode(cvode_mem, 2, yout, &t, CV_NORMAL);
  if(check_flag(&flag, (char *) "CVode", 1)) return(1);
  std::cout << "cvode 2\n";

  PrintOutput(t, yout, _size);
  std::cout << "cvode 3\n";

  return ans;
}

/*
 * f routine. Compute function f(t,y). 
 */

int Cvode::callf(realtype t, N_Vector Ny, N_Vector Nydot, void *user_data)
{
  DiffEq* ode = (DiffEq*) user_data;
                          std::cout << "callf 1\n";

  VectorMatrix My, Mydot;
  My = ode->getY();
                          std::cout << "callf 2\n";

  int dim_x = My.dimX();
                          std::cout << "callf 2\n";
  int dim_y = My.dimY();
                          std::cout << "callf 2\n";
  int dim_z = My.dimZ();
                          std::cout << "callf 2\n";
  const int dim_xy = dim_x * dim_y;

  std::cout << "matrixTest size: " << My.size() << "\n";
  std::cout << "matrixTest dimX: " << My.dimX() << "\n";
  std::cout << "matrixTest dimY: " << My.dimY() << "\n";
  std::cout << "matrixTest dimZ: " << My.dimZ() << "\n";

  {
    VectorMatrix::const_accessor Macc(My);

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

  getVectorMatrix(Ny,My);
                          std::cout << "callf 3\n";
  //getVectorMatrix(Nydot,Mydot);

  ode->diff(My,Mydot);
                          std::cout << "callf 4\n";

  //getN_Vector(My,Ny);
  getN_Vector(Mydot,Nydot);
                          std::cout << "callf 5\n";

  //Ith(Nydot,1) = RCONST(1);
  //Ith(Nydot,2) = t;
  //Ith(Nydot,3) = t * t;

  return(0);
}

matty::VectorMatrix Cvode::f(matty::VectorMatrix y)
{
  std::cout << "c++ VectorMatrix\n";
  return y;
}

/**
 * nvec muss mit der richtigen Größe initialisiert sein (N_VNew_Serial(3*size)).
 */
void Cvode::getN_Vector(const VectorMatrix& mat, N_Vector& nvec)
{
  int dim_x = mat.dimX();
  int dim_y = mat.dimY();
  int dim_z = mat.dimZ();
	const int dim_xy = dim_x * dim_y;

  VectorMatrix::const_accessor Macc(mat);

	for (int z=0; z<dim_z; ++z)
  for (int y=0; y<dim_y; ++y)
  for (int x=0; x<dim_x; ++x) {
    const int i = z*dim_xy + y*dim_x + x; // linear index of (x,y,z)
    Vector3d vec3 = Macc.get(i);
    Ith(nvec,3*i+1) = vec3[0];
    Ith(nvec,3*i+2) = vec3[1];
    Ith(nvec,3*i+3) = vec3[2];
  }
}

void Cvode::getVectorMatrix(const N_Vector& vec, VectorMatrix& mat)
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

    vec3[0] = Ith(vec,3*i+1);
    vec3[1] = Ith(vec,3*i+2);
    vec3[2] = Ith(vec,3*i+3);
    Macc.set(i, vec3);
  }
}

/*
 * Check function return value...
 * opt == 0 means SUNDIALS function allocates memory so check if
 *          returned NULL pointer
 * opt == 1 means SUNDIALS function returns a flag so check if
 *          flag >= 0
 * opt == 2 means function allocates memory so check if returned
 *          NULL pointer 
 */

int Cvode::check_flag(void *flagvalue, char *funcname, int opt)
{
  int *errflag;

  /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
  if (opt == 0 && flagvalue == NULL) {
    fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
        funcname);
    return(1); }

  /* Check if flag < 0 */
  else if (opt == 1) {
    errflag = (int *) flagvalue;
    if (*errflag < 0) {
      fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed with flag = %d\n\n",
          funcname, *errflag);
      return(1); }}

  /* Check if function returned NULL pointer - no memory allocated */
  else if (opt == 2 && flagvalue == NULL) {
    fprintf(stderr, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
        funcname);
    return(1); }

  return(0);
}

