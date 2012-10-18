#ifndef PYTHON_BYTEARRAY_H
#define PYTHON_BYTEARRAY_H

#include <Python.h>

#include <cstddef>

// PythonByteArray objects are copied and converted to Python bytearray objects when
// they are returned from a function/method.
// (see typemap defined in PythonByteArray.i)
//
// Objects of PythonByteArrays are copyable, and the internal byte array
// is shared between instances.

class PythonByteArray
{
public:
	PythonByteArray();
	PythonByteArray(size_t length);
	~PythonByteArray();

	char *get();
	size_t getSize(); 

private:
	// Todo: Use C++11 smart pointers...
	struct SharedArray 
	{
		SharedArray(char *ptr) : ptr(ptr), ref_cnt(0)
		{
			ref_cnt = new int;
			ref_cnt[0] = 1;
		}

		SharedArray(const SharedArray &other) : ptr(other.ptr), ref_cnt(other.ref_cnt)
		{
			ref_cnt[0] += 1;
		}

		~SharedArray() 
		{
			ref_cnt[0] -= 1;
			if (ref_cnt[0] == 0) delete [] ptr;
		}

		SharedArray &operator=(const SharedArray &other)
		{
			ptr = other.ptr;
			ref_cnt = other.ref_cnt;
			ref_cnt[0] += 1;
			return *this;
		}

		char *ptr;
		int *ref_cnt;
	};

	size_t length;
	SharedArray arr;
};

#endif
