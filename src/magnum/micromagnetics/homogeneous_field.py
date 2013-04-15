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

class HomogeneousField(module.Module):

    def __init__(self, var_id, default_value = None):
        super(HomogeneousField, self).__init__()
        self.__var_id        = var_id
        self.__default_value = default_value

        # attribute names of the state
        self.__value_id = "_HomogeneousField%i__value" % id(self)
        self.__field_id = "_HomogeneousField%i__field" % id(self)

    def calculates(self):
        return [self.__var_id]

    def updates(self):
        return [self.__var_id]

    def properties(self):
        return {}

    def initialize(self, system):
        self.__mesh = system.mesh

    def calculate(self, state, id):
        if id == self.__var_id:
            if hasattr(state, self.__field_id) and getattr(state, self.__field_id): return getattr(state, self.__field_id)
            return self.__update_field(state)

        else:
            raise ValueError("HomogeneousField: Don't know how to calculate %s." % id)

    def update(self, state, id, val):
        if id == self.__var_id:
            setattr(state, self.__value_id, val)
            setattr(state, self.__field_id, None)

        else:
            raise ValueError("HomogeneousField: Don't know how to calculate %s." % id)

    def __update_field(self, state):
        # Get value of homogeneous field
        if hasattr(state, self.__value_id):
            value = getattr(state, self.__value_id)
        else:
            if not self.__default_value: raise ValueError("HomogeneousField: Can't initialize field '%s' because no initial value is given!" % self.__var_id)
            value = self.__default_value

        # Create field (Field or VectorField) from value
        try:
            if isinstance(value, numbers.Number):
                value = float(value)
                field = Field(self.__mesh)
                field.fill(value)
            elif hasattr(value, "__iter__") and len(value) == 3:
                value = tuple(map(float, value))
                field = VectorField(self.__mesh)
                field.fill(value)
            else:
                raise ValueError
        except ValueError:
            raise ValueError("HomogeneousField: Expected scalar value or 3-tuple of scalar values for the 'value' (second) parameter")

        # Store value and field in state
        setattr(state, self.__value_id, value)
        setattr(state, self.__field_id, field)
        return field

class HomogeneousCurrent(HomogeneousField):
    def __init__(self, default_value=None):
        super(HomogeneousCurrent, self).__init__("j", default_value)
