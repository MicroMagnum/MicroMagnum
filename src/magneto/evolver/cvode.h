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

#include "matrix/matty.h"
#include <vector>
#include "/usr/local/include/cvode/cvode.h" // TODO relative path

#define Y1    RCONST(1.0)      /* initial y components */
#define Y2    RCONST(0.0)
#define Y3    RCONST(0.0)
#define T0    RCONST(0.0)      /* initial time           */
#define Ith(v,i)    NV_Ith_S(v,i-1)       /* Ith numbers components 1..NEQ */


// TODO
int cvode_test();

typedef struct {
    realtype a,b,c,d;
} *UserData;

static int f(realtype t, N_Vector y, N_Vector ydot, void *user_data);

static int check_flag(void *flagvalue, char *funcname, int opt);
