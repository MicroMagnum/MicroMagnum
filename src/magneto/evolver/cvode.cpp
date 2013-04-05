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

Cvode::Cvode(VectorMatrix &My, VectorMatrix &Mydot)
  : _My(My), _Mydot(Mydot)
{
    std::cout << "Konstruktor Anfang\n";
  _size  = My.size();
    std::cout << "size: " << _size << "\n";

    std::cout << "Konstruktor 1\n";
  _Ny = N_VNew_Serial(_size);
  _Nydot = N_VNew_Serial(_size);
    std::cout << "Konstruktor 2\n";
  getN_Vector(_My, _Ny);
    std::cout << "Konstruktor 3\n";
  getN_Vector(_Mydot, _Ny); //TODO remove
    std::cout << "Konstruktor 4\n";
  //_Nydot = N_VNew_Serial(_size);
    std::cout << "Konstruktor 5\n";
  getN_Vector(_Mydot, _Nydot);
    std::cout << "Konstruktor 6\n";

  _abstol = N_VNew_Serial(_size);
    std::cout << "Konstruktor 7\n";
  _reltol = 0.1;

  std::cout << "size: " << _size << std::endl;
  for (int i=1; i<=_size; ++i)
  {
    Ith(_abstol,i) = 0.1;
  }
}

static void PrintOutput(realtype t, realtype y1, realtype y2, realtype y3)
{
  std::cout << "t = " << t << ", y1 = " << y1 << ", y2 = " << y2 << ", y3 = " << y3 << "\n";
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
        //std::cout << std::endl;
      }
      //std::cout << std::endl;
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
  return 1; // TODO remove
  realtype t;
  N_Vector yout;
  void *cvode_mem;
  int flag, ans;
  VectorMatrix My, Mydot;
  DiffEq user_data;
  //user_data.diff(My,Mydot);

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
  flag = CVodeSetUserData(cvode_mem, &user_data);
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

  flag = CVode(cvode_mem, 2, yout, &t, CV_NORMAL);
  if(check_flag(&flag, (char *) "CVode", 1));

  PrintOutput(t, Ith(yout,1), Ith(yout,2), Ith(yout,3));

  return ans;
}

/*
 * f routine. Compute function f(t,y). 
 */

static int Cvode::callf(realtype t, N_Vector Ny, N_Vector Nydot, void *user_data)
{
  DiffEq* ode = (DiffEq*) user_data;

  matty::VectorMatrix My, Mydot;
  ode->diff(My,Mydot);

  Ith(Nydot,1) = RCONST(1);
  Ith(Nydot,2) = t;
  Ith(Nydot,3) = t * t;

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
void Cvode::getN_Vector(VectorMatrix mat, N_Vector& nvec)
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

void Cvode::getVectorMatrix(N_Vector vec, VectorMatrix& mat)
{
  int dim_x = mat.dimX();
  int dim_y = mat.dimY();
  int dim_z = mat.dimZ();
  //int size  = mat.size();
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

