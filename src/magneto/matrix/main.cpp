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

