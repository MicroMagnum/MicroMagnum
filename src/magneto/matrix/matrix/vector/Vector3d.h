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

#ifndef MATTY_VECTOR3D_H
#define MATTY_VECTOR3D_H

#include "config.h"

#include <ostream>
#include <cmath>
#include <cassert>

namespace matty {

/** @file */ 

/**
 * Vector with three elements.
 */
struct Vector3d
{
	/**
	 * Constructor. Initializes all elements to zero.
	 */
	inline Vector3d();
	
	/**
	 * Constructor with element initialization.
	 */
	inline Vector3d(double x, double y, double z);

	/**
	 * Constructor with equal x,y,z element initialization.
	 */
	inline explicit Vector3d(double xyz);

	/**
	 * Default destructor.
	 */
	inline ~Vector3d();

	/**
	 * Default copy constructor.
	 */
	inline Vector3d(const Vector3d &other);

	/**
	 * Get component, c = 0, 1, or 2.
	 */
	inline double &operator[](int c)
	{
		switch (c) {
			case 0: return x;
			case 1: return y;
			case 2: return z;
		}

		assert(0);
		static double foo = 0.0; return foo; // supress compiler warning...
	}

	inline const double &operator[](int c) const
	{
		switch (c) {
			case 0: return x;
			case 1: return y;
			case 2: return z;
		}

		assert(0);
		static double foo = 0.0; return foo; // supress compiler warning...
	}

	/**
	 * Default assignment operator.
	 */
	inline Vector3d &operator=(const Vector3d &other);

	/**
	 * Adds the vector 'other' to this vector.
	 */
	inline Vector3d &operator+=(const Vector3d &other);

	/**
	 * Substracts the vector 'other' from this vector.
	 */
	inline Vector3d &operator-=(const Vector3d &other);

	/**
	 * Copy contents of the vector 'other' to this vector.
	 */
	inline void assign(Vector3d &other);

	/**
	 * Explicitly assign the elements of this vector.
	 */
	inline void assign(double x, double y, double z);

	/**
	 * Return the length of this vector.
	 */
	inline double abs() const;

	/**
	 * Return the squared length of this vector, faster than abs().
	 */
	inline double abs_squared() const;

	/**
	 * Normalize this vector to length 'norm'.
	 */
	inline void normalize(double norm = 1.0);

	double x, y, z; /** the elements of the vector */
};

/**
 * Returns the dot/scalar product of the vectors 'lhs' and 'rhs'.
 */
inline double dot(const Vector3d &lhs, const Vector3d &rhs)
{
	return lhs.x*rhs.x + lhs.y*rhs.y + lhs.z*rhs.z;
}

/**
 * Returns a vector pointing into the same direction with length 'len'.
 */
inline Vector3d normalize(const Vector3d &lhs, double len = 1.0)
{
	Vector3d res(lhs);
	res.normalize(len);
	return res;
}

/**
 * Returns the cross product of the vectors lhs and rhs.
 */
inline Vector3d cross(const Vector3d &lhs, const Vector3d &rhs)
{
	return Vector3d(
		lhs.y*rhs.z - lhs.z*rhs.y,
		lhs.z*rhs.x - lhs.x*rhs.z,
		lhs.x*rhs.y - lhs.y*rhs.x
	);
}

/**
 * Returns the result of scaling the vector lhs with the scalar rhs.
 */
inline Vector3d operator*(const Vector3d &lhs, const double &rhs)
{
	return Vector3d(
		lhs.x * rhs,
		lhs.y * rhs,
		lhs.z * rhs
	);
}

/**
 * Returns the result of scaling the vector rhs with the scalar lhs.
 */
inline Vector3d operator*(const double lhs, const Vector3d &rhs)
{
	return Vector3d(
		rhs.x * lhs,
		rhs.y * lhs,
		rhs.z * lhs
	);
}

/**
 * Returns the result of dividing the elements of the vector lhs by the scalar rhs.
 */
inline Vector3d operator/(const Vector3d &lhs, const double &rhs)
{
	return Vector3d(
		lhs.x / rhs,
		lhs.y / rhs,
		lhs.z / rhs
	);
}

/**
 * Returns the result of summing the vectors lhs and rhs.
 */
inline Vector3d operator+(const Vector3d &lhs, const Vector3d &rhs)
{
	return Vector3d(
		lhs.x + rhs.x,
		lhs.y + rhs.y,
		lhs.z + rhs.z
	);
}

/**
 * Returns the result of subtracting the vectors lhs from rhs.
 */
inline Vector3d operator-(const Vector3d &lhs, const Vector3d &rhs)
{
	return Vector3d(
		lhs.x - rhs.x,
		lhs.y - rhs.y,
		lhs.z - rhs.z
	);
}

/**
 * Compares the vector lhs and rhs element-wise.
 */
inline bool operator==(const Vector3d &lhs, const Vector3d &rhs)
{
	return (lhs.x == rhs.x) && (lhs.y == rhs.y) && (lhs.z == rhs.z);
}

/**
 * Compares the vector lhs and rhs element-wise.
 */
inline bool operator!=(const Vector3d &lhs, const Vector3d &rhs)
{
	return !(lhs == rhs);
}

/**
 * Output to stream.
 */
std::ostream &operator<<(std::ostream &out, const Vector3d &vec);

////////////////////////////////////////////////////////////////////////////////

Vector3d::Vector3d()
	: x(0), y(0), z(0)
{
}

Vector3d::Vector3d(double x, double y, double z)
	: x(x), y(y), z(z)
{
}

Vector3d::Vector3d(double xyz)
	: x(xyz), y(xyz), z(xyz)
{
}

Vector3d::~Vector3d()
{
}

Vector3d::Vector3d(const Vector3d &other)
{
	assign(other.x, other.y, other.z);
}

Vector3d &Vector3d::operator=(const Vector3d &other)
{
	assign(other.x, other.y, other.z);
	return *this;
}

Vector3d &Vector3d::operator+=(const Vector3d &other)
{
	x += other.x; y += other.y; z += other.z;
	return *this;
}

Vector3d &Vector3d::operator-=(const Vector3d &other)
{
	x -= other.x; y -= other.y; z -= other.z;
	return *this;
}

double Vector3d::abs() const 
{ 
	return std::sqrt(abs_squared()); 
}

double Vector3d::abs_squared() const 
{ 
	return dot(*this, *this);
}

void Vector3d::normalize(double norm)
{
	const double len = abs();
	if (len == 0.0) {
		return;
	}

	const double factor = norm / len;
	x *= factor;
	y *= factor;
	z *= factor;
}

void Vector3d::assign(Vector3d &other)
{
	assign(other.x, other.y, other.z);
}

void Vector3d::assign(double x, double y, double z)
{
	this->x = x; this->y = y; this->z = z;
}

/*
template <class accessor>
Vector3d vector_get(accessor &acc, int nth)
{
	return Vector3d(acc.linearGet(nth,0), acc.linearGet(nth,1), acc.linearGet(nth,2));
}

template <class accessor>
Vector3d vector_get(accessor &acc, int x, int y, int z)
{
	return Vector3d(acc.get(x,y,z,0), acc.get(x,y,z,1), acc.get(x,y,z,2));
}

template <class accessor>
void vector_set(accessor &acc, int nth, const Vector3d &vec)
{
	acc.linearSet(nth,0,vec.x);
	acc.linearSet(nth,1,vec.y);
	acc.linearSet(nth,2,vec.z);
}

template <class accessor>
void vector_set(accessor &acc, int x, int y, int z, const Vector3d &vec)
{
	acc.set(x,y,z,0,vec.x);
	acc.set(x,y,z,1,vec.y);
	acc.set(x,y,z,2,vec.z);
}
*/

} // ns

#endif

