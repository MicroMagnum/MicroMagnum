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

#include "config.h"
#include "cvode.h"
#include "Magneto.h"
#include "/usr/local/include/cvode/cvode.h" // TODO relative path
//extern "C" {
//  #include "/usr/local/include/nvector/nvector_parallel.h"
//}
#include "/usr/local/include/nvector/nvector_serial.h"
#include "/usr/local/include/cvode/cvode_band.h"
#include "/usr/local/include/sundials/sundials_types.h"

// TODO implementation
int cvode_test() {
  realtype t;
  N_Vector y, ydot;
  void *cvode_mem;
  UserData user_data;
  int flag, ans;

  y = ydot = NULL;
  cvode_mem = NULL;
  user_data = NULL;

  y = N_VNew_Serial(3);
  ydot = N_VNew_Serial(3);

  Ith(y,1) = Y1;
  Ith(y,2) = Y2;
  Ith(y,3) = Y3;

  user_data = (UserData) malloc(sizeof *user_data);  /* Allocate data memory */

  /* Call CVodeCreate to create the solver memory and specify the 
   * Backward Differentiation Formula and the use of a Newton iteration */
  cvode_mem = CVodeCreate(CV_BDF, CV_NEWTON);
  if (check_flag((void *)cvode_mem, "CVodeCreate", 0)) return(1);

  /* Call CVodeInit to initialize the integrator memory and specify the
   * user's right hand side function in y'=f(t,y), the inital time T0, and
   * the initial dependent variable vector y. */
  flag = CVodeInit(cvode_mem, f, T0, y);
  if (check_flag(&flag, "CVodeInit", 1)) return(1);

  /* Set the pointer to user-defined data */
  //flag = CVodeSetUserData(cvode_mem, user_data);
  //if(check_flag(&flag, "CVodeSetUserData", 1)) return(1);

  // ans = CVRhsFn(t, y, ydot, user_data);

  return ans;
}

/*
 * f routine. Compute function f(t,y). 
 */

static int f(realtype t, N_Vector y, N_Vector ydot, void *user_data)
{
  realtype y1, y2, y3, yd1, yd3;

  y1 = Ith(y,1); y2 = Ith(y,2); y3 = Ith(y,3);

  yd1 = Ith(ydot,1) = RCONST(-0.04)*y1 + RCONST(1.0e4)*y2*y3;
  yd3 = Ith(ydot,3) = RCONST(3.0e7)*y2*y2;
  Ith(ydot,2) = -yd1 - yd3;

  return(0);
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

static int check_flag(void *flagvalue, char *funcname, int opt)
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

