# Copyright 2012, 2013 by the Micromagnum authors.
#
# This file is part of MicroMagnum.
#
# MicroMagnum is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MicroMagnum is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with MicroMagnum.  If not, see <http://www.gnu.org/licenses/>.

import magnum.module as module
from magnum.mesh import VectorField, Field
from magnum.logger import logger
from math import sin

# AlternatingHomogeneousField
class AlternatingField(module.Module):
    def __init__(self, var_id):
        super(AlternatingField, self).__init__()

        # Parameters
        self.__offs = var_id + "_offs"
        self.__amp  = var_id + "_amp"
        self.__freq = var_id + "_freq"
        self.__phase= var_id + "_phase"
        self.__func = var_id + "_fn"

        setattr(self, self.__offs,  (0.0, 0.0, 0.0))
        setattr(self, self.__amp,   (0.0, 0.0, 0.0))
        setattr(self, self.__freq,  (0.0, 0.0, 0.0))
        setattr(self, self.__phase, (0.0, 0.0, 0.0))
        setattr(self, self.__func,  None)

        # Generated model variables
        self.__var = var_id

    def calculates(self):
        return [self.__var]

    def params(self):
        return [self.__offs, self.__amp, self.__freq, self.__phase, self.__func]

    def properties(self):
        return {}

    def initialize(self, system):
        self.system = system
        logger.info("%s: Providing model variable %s, parameters are %s" % (self.name(), self.__var, ", ".join(self.params())))

    def calculate(self, state, id):
        if id == self.__var:
            # Get parameters...
            offs  = getattr(self, self.__offs)
            amp   = getattr(self, self.__amp)
            freq  = getattr(self, self.__freq)
            phase = getattr(self, self.__phase)
            fn    = getattr(self, self.__func)

            # Calculate field 'A'.
            t = state.t
            if fn: # with user function
                if any(x != (0.0, 0.0, 0.0) for x in (amp, freq, phase, offs)):
                    raise ValueError("AlternatingField.calculates: If %s is defined, the parameters %s, %s, %s and %s must be zero vectors, i.e. (0.0, 0.0, 0.0)" % (self.__func, self.__offs, self.__amp, self.__freq, self.__phase))
                # call user function
                A = fn(t)
            else:
                # with 'offs', 'amp', 'freq', 'phase' parameters
                A = (offs[0] + amp[0] * sin(t * freq[0] + phase[0]),
                     offs[1] + amp[1] * sin(t * freq[1] + phase[1]),
                     offs[2] + amp[2] * sin(t * freq[2] + phase[2]))

            # Convert 3-vector to VectorField if necessary.
            if isinstance(A, tuple):
                tmp = A; A = VectorField(self.system.mesh); A.fill(tmp)

            # Return field 'A'
            return A

        else:
            raise KeyError("AlternatingField.calculates: Can't calculate %s", id)
