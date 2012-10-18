#ifndef CALLBACK_H
#define CALLBACK_H

#include <Python.h>
#include <string>

class PythonCallable
{
	PyObject *func;
public:
	PythonCallable();
	PythonCallable(PyObject *func);
	PythonCallable(const PythonCallable &other);
	PythonCallable &operator=(PythonCallable other);
	~PythonCallable();

	void call();
	void call(int i, const std::string &str);
};

#endif

