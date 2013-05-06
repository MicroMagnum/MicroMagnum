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

#include "config.h"
#include "PythonCallable.h"

#include <algorithm>

#define DO_LOCK_GIL 0

PythonCallable::PythonCallable()
	: func(0)
{
}

PythonCallable::PythonCallable(PyObject *func)
	: func(func)
{
	Py_XINCREF(func);
}

PythonCallable::PythonCallable(const PythonCallable &other)
	: func(other.func)
{
	Py_XINCREF(func);
}

PythonCallable &PythonCallable::operator=(PythonCallable other)
{
	std::swap(func, other.func);
	return *this;
}

PythonCallable::~PythonCallable()
{
	Py_XDECREF(func);
}

#if DO_LOCK_GIL
static PyGILState_STATE gstate;

static void lock_gil()
{
	gstate = PyGILState_Ensure();
}

static void unlock_gil()
{
	PyGILState_Release(gstate);
}
#endif

void PythonCallable::call()
{
#if DO_LOCK_GIL
	lock_gil();
#endif

	PyObject *arglist = Py_BuildValue("()");
	PyObject *result = PyEval_CallObject(func, arglist);

	Py_DECREF(arglist);
	Py_XDECREF(result);

#if DO_LOCK_GIL
	unlock_gil();
#endif
}

void PythonCallable::call(int i, const std::string &str)
{
#if DO_LOCK_GIL
	lock_gil();
#endif

	PyObject *arglist = Py_BuildValue("(i,s)", i, str.c_str());
	PyObject *result = PyEval_CallObject(func, arglist);

	Py_DECREF(arglist);
	Py_XDECREF(result);

#if DO_LOCK_GIL
	unlock_gil();
#endif
}

