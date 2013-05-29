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

#define Ith(v,i)    NV_Ith_S(v,i)       /* Ith numbers components 1..NEQ */

//using namespace matty;
class DiffEq {
  public:
    /*
     * @param My  Initial state VectorMatrix
     */
    DiffEq(VectorMatrix &My);
    virtual ~DiffEq();

    /*
     * Function called to calculate micromagnetics.
     *
     * @param Ny     current State N_Vector
     * @param Nydot  result        N_Vector
     * @param t      current time  realtype
     */
    void diffN(const N_Vector& Ny, N_Vector& Nydot, realtype t);

    /*
     * Director class, calculates micromagnetics in python.
     * returns a copy of result. Inefficient!
     */
    virtual VectorMatrix diff(const VectorMatrix &My);

    /*
     * Director class, calculates micromagnetics in python.
     * 
     * @param Mydot result
     */
    virtual void diffX(const VectorMatrix &My, VectorMatrix &Mydot, double t);

    /*
     * Saves current result to state
     */
    virtual void saveState(VectorMatrix yout);

    /*
     * converts N_Vector and saves State
     */
    void saveStateC(N_Vector yout);

    /*
     * Returns a reference of initial VectorMatrix.
     */
    virtual VectorMatrix getY();

    /*
     * Saves Time to step in state
     */
    virtual void saveTime(double t);

    /*
     * increases substep in state
     */
    virtual void substep();

    /*
     * converts N_Vector to VectorMatrix
     */
    static void getVectorMatrix(const N_Vector& vec, VectorMatrix& mat);

    /*
     * converts VectorMatrix to N_Vector
     */
    static void getN_Vector(const matty::VectorMatrix& vec, N_Vector& nvec);

    /*
     * Print methods
     */
    static void printVectorMatrix(const VectorMatrix& mat);
    void printN_Vector(const N_Vector& nvec);
    void printOutput(const realtype &t, const N_Vector& nvec);

    /*
     * tests matrix conversion.
     * TODO move to Magnum tests
     */
    static void matrixTest(VectorMatrix mat);

    /*
     * size of initial VectorMatrix. Number of scalars.
     */
    int size();

    /*
     * Object variables
     */
    VectorMatrix &_My, &_Mydot;
};
#endif
