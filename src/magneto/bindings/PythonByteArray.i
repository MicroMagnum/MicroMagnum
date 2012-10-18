
%typemap(out) (PythonByteArray) 
{
        // Note: PyByteArray_FromStringAndSize does an internal copy of the data.
        $result = PyByteArray_FromStringAndSize($1.get(), $1.getSize());
} 

