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
import magnum.magneto as magneto

from magnum.mesh import VectorField, Field
from magnum.micromagnetics.constants import MU0

class ExchangeField(module.Module):
    def __init__(self):
        super(ExchangeField, self).__init__()

    def calculates(self):
        return ["H_exch", "E_exch"]

    def params(self):
        return ["A"]

    def properties(self):
        return {'EFFECTIVE_FIELD_TERM': "H_exch", 'EFFECTIVE_FIELD_ENERGY': "E_exch"}

    def initialize(self, system):
        self.system = system
        self.A = Field(self.system.mesh); self.A.fill(0.0)

    def calculate(self, state, id):
        cache = state.cache

        if id == "H_exch":
            if hasattr(cache, "H_exch"): return cache.H_exch
            H_exch = cache.H_exch = VectorField(self.system.mesh)

            magneto.exchange(state.Ms, state.A, state.M, H_exch)
            return H_exch

        elif id == "E_exch":
            return -MU0 / 2.0 * self.system.mesh.cell_volume * state.M.dotSum(state.H_exch)

        else:
            raise KeyError("ExchangeField.calculate: Can't calculate %s", id)
