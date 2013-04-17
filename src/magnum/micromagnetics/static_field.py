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
from magnum.mesh import Field, VectorField
import numbers

class StaticField(module.Module):

    def __init__(self, var_id):
        super(StaticField, self).__init__()
        self.__var_id      = var_id
        self.__default_val = default_val
        self.__field       = None

    def calculates(self):
        return [self.__var_id]

    def params(self):
        return [self.__var_id]

    def properties(self):
        return {}

    def initialize(self, system):
        self.system = system

    def calculate(self, state, id):
        if id == self.__var_id:
            fld = self.__field
        else:
            raise ValueError("%s: Don't know how to calculate %s." % (self.name(), id))

    def set_param(self, id, val):
        if id == self.__var_id:
            if isinstance(val, numbers.Number):
              fld = Field(self.system.mesh)
              fld.fill(val)
            elif hasattr(value, "__iter__") and len(value) == 3:
                val = tuple(map(float, val))
                field = VectorField(self.system.mesh)
                field.fill(val)
            elif isinstance(val, Field):
                fld = Field(self.system.mesh)
                fld.assign(val)
            elif isinstance(val, VectorField):
                fld = VectorField(self.system.mesh)
                fld.assign(val)
            else:
                raise ValueError
            self.__field = fld
        else:
            raise ValueError("%s: Don't know how to update %s." % (self.name(), id))
