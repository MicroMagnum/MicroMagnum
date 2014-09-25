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

/* Convert from C -> Python */
%typemap(out) (PythonByteArray) 
{
        // Note: PyByteArray_FromStringAndSize does an internal copy of the data.
        $result = PyByteArray_FromStringAndSize($1.get(), $1.getSize());
        /*printf("Out: ByteArray: (%lu) (%p) (%s)\n", $1.getSize(), $1.get(), $1.get()); */
} 

/* Convert from Python -> C */
%typemap(in) (PythonByteArray) 
{
        /* printf("In: ByteArray: (%d) (%d) (%lu)\n", PyByteArray_Check($input), PyByteArray_CheckExact($input), PyByteArray_Size($input)); */
        char *str = PyByteArray_AsString($input);
        const size_t len = PyByteArray_Size($input);
        $1 = PythonByteArray(len); /* FloB: new object is created with length 'len', although original
                                            object existed, but with unknown length. */
        memcpy($1.get(), str, len);
} 
