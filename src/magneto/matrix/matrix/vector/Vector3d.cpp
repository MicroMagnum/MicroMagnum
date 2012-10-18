#include "config.h"
#include "Vector3d.h"

namespace matty {

std::ostream &operator<<(std::ostream &out, const Vector3d &vec)
{
	out << "(" << vec.x << "," << vec.y << "," << vec.z << ")";
	return out;
}

} // ns
