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

from magnum.mesh import VectorField
from magnum.module import Module

from magnum.micromagnetics.stray_field_calculator import DemagTensorField, StrayFieldCalculator
from magnum.micromagnetics.constants import MU0

class StrayField(Module):
    def __init__(self, method = "tensor"):
        super(StrayField, self).__init__()
        self.method = method
        self.padding = DemagTensorField.PADDING_ROUND_4

    def calculates(self):
        return ["H_stray", "E_stray"]

    def properties(self):
        return {'EFFECTIVE_FIELD_TERM': "H_stray",
                'EFFECTIVE_FIELD_ENERGY': "E_stray"}

    def initialize(self, system):
        self.system = system
        self.calculator = StrayFieldCalculator(system.mesh, self.method, self.padding)

    def calculate(self, state, id):
        cache = state.cache

        if id == "H_stray":
            if hasattr(cache, "H_stray"): return cache.H_stray
            H_stray = cache.H_stray = VectorField(self.system.mesh)
            self.calculator.calculate(state.M, H_stray)
            return H_stray

        elif id == "E_stray":
            if hasattr(cache, "E_stray"): return cache.E_stray
            E_stray = cache.E_stray = -MU0 / 2.0 * self.system.mesh.cell_volume * state.M.dotSum(state.H_stray)
            return E_stray

        else:
            raise KeyError("StrayField.calculate: %s" % id)
