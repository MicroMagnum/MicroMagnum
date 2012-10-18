#ifndef MATTY_SHAPE_H
#define MATTY_SHAPE_H

#include "config.h"
#include <vector>

namespace matty {

class Shape
{
public:
	Shape();
	Shape(int x0);
	Shape(int x0, int x1);
	Shape(int x0, int x1, int x2);
	Shape(int x0, int x1, int x2, int x3);
	~Shape();

	int operator[](int d) const { return getDim(d); }

	// get linear index (unsafe! they assume that shape rank is high enough)
	int getLinIdx(int x0) const;
	int getLinIdx(int x0, int x1) const;
	int getLinIdx(int x0, int x1, int x2) const;
	int getLinIdx(int x0, int x1, int x2, int x3) const;

	// get dimensions
	int getDim(int d) const { return dims[d]; }
	const std::vector<int> &getDims() const { return dims; }

	// get strides
	int getStride(int d) const { return strides[d]; }
	const std::vector<int> &getStrides() const { return strides; }

	int getRank() const { return dims.size(); }
	int getNumEl() const;

	bool sameDims(const Shape &other) const;

private:
	std::vector<int> dims;
	std::vector<int> strides;

	void init(int *beg, int *end);
};

inline int Shape::getLinIdx(int x0) const
{
	return strides[0] * x0;
}

inline int Shape::getLinIdx(int x0, int x1) const
{
	return strides[0] * x0 + strides[1] * x1;
}

inline int Shape::getLinIdx(int x0, int x1, int x2) const
{
	return strides[0] * x0 + strides[1] * x1 + strides[2] * x2;
}

inline int Shape::getLinIdx(int x0, int x1, int x2, int x3) const
{
	return strides[0] * x0 + strides[1] * x1 + strides[2] * x2 + strides[3] * x3;
}

} //ns

#endif
