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

#ifndef diffeq_h
#define diffeq_h

#include "matrix/matty.h"
#include <nvector/nvector_serial.h>

#define Ith(v,i)    NV_Ith_S(v,i-1)       /* Ith numbers components 1..NEQ */

//using namespace matty;
class DiffEq {
  public:
    DiffEq(VectorMatrix &My);
    //virtual void diff(const VectorMatrix &My, VectorMatrix &Mydot);
    virtual VectorMatrix diff(const VectorMatrix &My);
    virtual void diffX(const VectorMatrix &My, VectorMatrix &Mydot);
    void diffN(const N_Vector& Ny, N_Vector& Nydot);
    virtual VectorMatrix getY();

    VectorMatrix &_My, &_Mydot;
    //N_Vector& Ny, Nydot;

    static void getVectorMatrix(const N_Vector& vec, VectorMatrix& mat);
    static void getN_Vector(const matty::VectorMatrix& vec, N_Vector& nvec);
    static void printVectorMatrix(const VectorMatrix& mat);
    void printN_Vector(const N_Vector& nvec);
    void printOutput(const realtype &t, const N_Vector& nvec);
    int size();

    static void matrixTest(VectorMatrix mat);
};
#endif
