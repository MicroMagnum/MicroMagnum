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

class AnisotropyField(module.Module):
    def __init__(self):
        super(AnisotropyField, self).__init__()

    def calculates(self):
        return ["H_aniso", "E_aniso"]

    def params(self):
        return ["k_uniaxial", "k_cubic", "axis1", "axis2"]

    def properties(self):
        return {'EFFECTIVE_FIELD_TERM': "H_aniso", 'EFFECTIVE_FIELD_ENERGY': "E_aniso"}

    def initialize(self, system):
        self.system = system
        self.k_uniaxial = Field(self.system.mesh); self.k_uniaxial.fill(0.0)
        self.k_cubic = Field(self.system.mesh); self.k_cubic.fill(0.0)
        self.axis1 = VectorField(self.system.mesh); self.axis1.fill((0.0, 0.0, 0.0))
        self.axis2 = VectorField(self.system.mesh); self.axis2.fill((0.0, 0.0, 0.0))

    def on_param_update(self, id):
        if id in self.params() + ["Ms"]:
            axis1, axis2 = self.axis1, self.axis2
            k_uni, k_cub = self.k_uniaxial, self.k_cubic
            Ms = self.system.get_param("Ms")

            def compute_none(state, H_aniso):
                H_aniso.fill((0.0, 0.0, 0.0))
                return 0.0

            def compute_uniaxial(state, H_aniso):
                return magneto.uniaxial_anisotropy(axis1, k_uni, Ms, state.M, H_aniso)

            def compute_cubic(state, H_aniso):
                return magneto.cubic_anisotropy(axis1, axis2, k_cub, Ms, state.M, H_aniso)

            def compute_uniaxial_and_cubic(state, H_aniso):
                tmp = VectorField(self.system.mesh)
                E0 = magneto.uniaxial_anisotropy(axis1, k_uni, Ms, state.M, tmp)
                E1 = magneto.cubic_anisotropy(axis1, axis2, k_cub, Ms, state.M, H_aniso)
                state.cache.E_aniso_sum = E0 + E1
                H_aniso.add(tmp)

            fns = {(False, False): compute_none,
                   ( True, False): compute_uniaxial,
                   (False,  True): compute_cubic,
                   ( True,  True): compute_uniaxial_and_cubic}

            have_uni = not (k_uni.isUniform() and k_uni.uniform_value == 0.0)
            have_cub = not (k_cub.isUniform() and k_cub.uniform_value == 0.0)
            self.__compute_fn = fns[have_uni, have_cub]

    def calculate(self, state, id):
        cache = state.cache
        if id == "H_aniso":
            if hasattr(cache, "H_aniso"): return cache.H_aniso
            H_aniso = cache.H_aniso = VectorField(self.system.mesh)
            cache.E_aniso_sum = self.__compute_fn(state, H_aniso)
            return H_aniso

        elif id == "E_aniso":
            if not hasattr(cache, "E_aniso_sum"):
                self.calculate(state, "H_aniso")
            return cache.E_aniso_sum * self.system.mesh.cell_volume

        else:
            raise KeyError("AnisotropyField.calculate: Can't calculate %s", id)
