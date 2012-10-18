
%typemap(in) PythonCallable {
        if (!PyCallable_Check($input)) {
                PyErr_SetString(PyExc_TypeError, "Need a callable object!");
                return 0;
        }
        {
                PythonCallable tmp($input);
                $1 = tmp;
        }
}

