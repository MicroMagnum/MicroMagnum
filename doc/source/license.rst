=======
License
=======

MicroMagnum is free software and is licensed under the General
Public License 3 (GPL3). For details, please see the file COPYING
in the distribution package. The GPL3 license is also available 
online at http://www.gnu.org/licenses/gpl.txt .

Used libraries
==============

Some parts of the code were copied and adapted from the Object-Oriented
Micromagnetic Framework (OOMMF), which is released as public domain code
by Michael Donahue and Don Porter:

- the implementation of Newells formulas in src/magneto/mmm/demag/demag_coeff.h
- the cubic anisotropy field computation in src/magneto/mmm/anisotropy/anisotropy_cpu.cpp

The VTK output routines were adapted from the PyEVtk library at
https://bitbucket.org/pauloh/pyevtk by Paulo A. Herrera. The copyright
notice of the PyEVtk package is shown below:

.. code-block:: text

   ***********************************************************************************
   * Copyright 2010 Paulo A. Herrera. All rights reserved.                           * 
   *                                                                                 *
   * Redistribution and use in source and binary forms, with or without              *
   * modification, are permitted provided that the following conditions are met:     *
   *                                                                                 *
   *  1. Redistributions of source code must retain the above copyright notice,      *
   *  this list of conditions and the following disclaimer.                          *
   *                                                                                 *
   *  2. Redistributions in binary form must reproduce the above copyright notice,   *
   *  this list of conditions and the following disclaimer in the documentation      *
   *  and/or other materials provided with the distribution.                         *
   *                                                                                 *
   * THIS SOFTWARE IS PROVIDED BY PAULO A. HERRERA ``AS IS'' AND ANY EXPRESS OR      *
   * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF    *
   * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO      *
   * EVENT SHALL <COPYRIGHT HOLDER> OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,        *
   * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,  *
   * BUT NOT LIMITED TO, PROCUREMEN OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,    *
   * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY           *
   * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING  *
   * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS              *
   * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                    *
   ***********************************************************************************


