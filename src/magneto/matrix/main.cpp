/*
 * Copyright 2012, 2013 by the Micromagnum authors.
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

#include "matty.h"

#include <iostream>
using namespace std;

using namespace matty;

void test()
{
	Matrix M1 = zeros(Shape(100, 100, 100));
	Matrix M2 = zeros(Shape(100, 100, 100));
	Matrix M3 = zeros(Shape(100, 100, 100));
	M1.randomize();
	M2.randomize();
	M1.scale(100);
	M2.scale(50);
	M1.add(M2);
	cout << M1.average() << endl;
	cout << M1.maximum() << endl;

	matty::getDevice(0)->printReport(cout);

	Matrix::const_accessor M1_acc(M1);
	cout << M1_acc.at(12, 13, 14) << endl;
}

void test2()
{
	VectorMatrix M(Shape(100, 100, 1));
	M.randomize();

	Matrix S(Shape(100, 100, 1));
	S.randomize();
	S.scale(0.5);

	M.scale(Vector3d(0.0, 0.0, 1.0));

	cout << M.average() << endl;

}

int main()
{
	matty::initialize();
	test2();
	matty::deinitialize();
}

