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

from .alternating_field import AlternatingField
from .static_field import StaticField
from .constants import MU0

class AlternatingExternalField(AlternatingField):
    def __init__(self):
        super(AlternatingExternalField, self).__init__("H_ext")

    def calculates(self):
        return super(AlternatingExternalField, self).calculates() + ["E_ext"]

    def properties(self):
        p = super(AlternatingExternalField, self).properties()
        p.update({'EFFECTIVE_FIELD_TERM': "H_ext", 'EFFECTIVE_FIELD_ENERGY': "E_ext"})
        return p

    def calculate(self, state, id):
        if id == "E_ext":
            return -MU0 * self.system.mesh.cell_volume * state.M.dotSum(state.H_ext)
        else:
            return super(AlternatingExternalField, self).calculate(state, id)

ExternalField = AlternatingExternalField  # alias

class StaticExternalField(StaticField):
    def __init__(self, field_id="H_ext", energy_id="E_ext"):
        super(StaticExternalField, self).__init__(field_id)
        self.__field_id = var_id
        self.__energy_id = energy_id

    def calculates(self):
        return super(StaticExternalField, self).calculates() + [self.__energy_id]

    def properties(self):
        p = super(StaticExternalField, self).properties()
        p.update({'EFFECTIVE_FIELD_TERM': self.__var_id, 'EFFECTIVE_FIELD_ENERGY': self.__energy_id})
        return p

    def calculate(self, state, id):
        if id == self.__energy_id:
            return -MU0 * self.system.mesh.cell_volume * state.M.dotSum(getattr(state, self.__field_id))
        else:
            return super(StaticExternalField, self).calculate(state, id)
