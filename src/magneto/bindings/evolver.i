%{
#include "evolver/runge_kutta.h"
%}

struct ButcherTableau
{
	/*
	   a0 |
	   a1 | b10
	   a2 | b20 b21
	   a3 | b30 b31 b32
	   a4 | b40 b41 b42 b43
	   a5 | b50 b51 b52 b53 b54 
	   ---+-------------------------
              |  c0  c1  c2  c3  c4  c5
	      | ec0 ec1 ec2 ec3 ec4 ec5
	*/

	ButcherTableau(int num_steps);
	~ButcherTableau();

	void   setA (int i, double v);
	double getA (int i);
	void   setB (int i, int j, double v);
	double getB (int i, int j);
	void   setC (int i, double v);
	double getC (int i);
	void   setEC(int i, double v);
	double getEC(int i);

        int getNumSteps();
};

void rk_prepare_step(
	int step,
	double h,
	ButcherTableau &tab,

	const VectorMatrix &k0,
	const VectorMatrix &k1,
	const VectorMatrix &k2,
	const VectorMatrix &k3,
	const VectorMatrix &k4,
	const VectorMatrix &k5,

	const VectorMatrix &y,
	VectorMatrix &ytmp
);

void rk_combine_result(
	double h,
	ButcherTableau &tab,

	const VectorMatrix &k0,
	const VectorMatrix &k1,
	const VectorMatrix &k2,
	const VectorMatrix &k3,
	const VectorMatrix &k4,
	const VectorMatrix &k5,

	VectorMatrix &y,
	VectorMatrix &y_error
);

double rk_adjust_stepsize(int order, double h, double eps_abs, double eps_rel, const VectorMatrix &y, const VectorMatrix &y_error);
