#include "config.h"
#include "Shape.h"

#include <cstddef>

namespace matty {

Shape::Shape()
{
	int dims[1];
	init(dims, dims+0);
}

Shape::Shape(int dim_x)
{
	int dims_[] = {dim_x};
	init(dims_, dims_+1);
}

Shape::Shape(int dim_x, int dim_y)
{
	int dims_[] = {dim_x, dim_y};
	init(dims_, dims_+2);
}

Shape::Shape(int dim_x, int dim_y, int dim_z)
{
	int dims_[] = {dim_x, dim_y, dim_z};
	init(dims_, dims_+3);
}

Shape::Shape(int x0, int x1, int x2, int x3)
{
	int dims_[] = {x0, x1, x2, x3};
	init(dims_, dims_+4);
}

Shape::~Shape()
{
}

void Shape::init(int *beg, int *end)
{
	dims = std::vector<int>(beg, end);

	int s = 1;
	for (std::size_t i=0; i<dims.size(); ++i) {
		strides.push_back(s);
		s *= dims[i];
	}
}

int Shape::getNumEl() const
{
	int n = 1;
	for (int i=0; i<getRank(); ++i) {
		n *= getDim(i);
	}
	return n;
}

bool Shape::sameDims(const Shape &other) const
{
	if (getRank() != other.getRank()) return false;

	for (int i=0; i<getRank(); ++i) {
		if (getDim(i) != other.getDim(i)) return false;
	}
	return true;
}

} // ns
