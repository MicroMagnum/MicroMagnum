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

from magnum.logger import logger
from magnum.mesh import VectorField, Field

from .constants import GYROMAGNETIC_RATIO

import math

class LandauLifshitzGilbert(module.Module):
    def __init__(self, do_precess=True):
        super(LandauLifshitzGilbert, self).__init__()
        self.__do_precess = do_precess
        self.__valid_factors = False

    def calculates(self):
        return ["dMdt", "M", "H_tot", "E_tot", "deg_per_ns"]

    def updates(self):
        return ["M"]

    def params(self):
        return ["Ms", "alpha"]

    def on_param_update(self, id):
        if id in self.params():
            self.__valid_factors = False

    def initialize(self, system):
        self.system = system
        self.Ms = Field(system.mesh); self.Ms.fill(0.0)
        self.alpha = Field(system.mesh); self.alpha.fill(0.0)
        self.__valid_factors = False

        # Find other active modules
        self.field_terms = []
        self.field_energies = []
        self.llge_terms = []
        for mod in system.modules:
            prop = mod.properties()
            if 'EFFECTIVE_FIELD_TERM'   in prop.keys(): self.field_terms.append(prop['EFFECTIVE_FIELD_TERM'])
            if 'EFFECTIVE_FIELD_ENERGY' in prop.keys(): self.field_energies.append(prop['EFFECTIVE_FIELD_ENERGY'])
            if 'LLGE_TERM'              in prop.keys(): self.llge_terms.append(prop['LLGE_TERM'])

        logger.info("LandauLifshitzGilbert module configuration:")
        logger.info(" - H_tot = %s", " + ".join(self.field_terms) or "0")
        logger.info(" - E_tot = %s", " + ".join(self.field_energies) or "0")
        logger.info(" - dM/dt = %s", " + ".join(["LLGE(M, H_tot)"] + self.llge_terms) or "0")
        if not self.__do_precess: logger.info(" - Precession term is disabled")

    def calculate(self, state, id):
        if id == "M":
            return state.y
        elif id == "H_tot" or id == "H_eff":
            return self.calculate_H_tot(state)
        elif id == "E_tot" or id == "E_eff":
            return self.calculate_E_tot(state)
        elif id == "dMdt":
            return self.calculate_dMdt(state)
        elif id == "deg_per_ns":
            return self.calculate_deg_per_ns(state)
        else:
            raise KeyError(id)

    def update(self, state, id, value):
        if id == "M":
            logger.info("Assigning new magnetic state M")
            module.assign(state.y, value)
            state.y.normalize(state.Ms)   # XXX: Is this a good idea? (Solution: Use unit magnetization everywhere.)
            state.flush_cache()
        else:
            raise KeyError(id)

    def calculate_H_tot(self, state):
        if hasattr(state.cache, "H_tot"): return state.cache.H_tot
        H_tot = state.cache.H_tot = VectorField(self.system.mesh)

        H_tot.fill((0.0, 0.0, 0.0))
        for H_id in self.field_terms:
            H_i = getattr(state, H_id)
            H_tot.add(H_i)

        return H_tot

    def calculate_E_tot(self, state):
        if hasattr(state.cache, "E_tot"): return state.cache.E_tot
        state.cache.E_tot = sum(getattr(state, E_id) for E_id in self.field_energies) # calculate sum of all registered energy terms
        return state.cache.E_tot

    def calculate_dMdt(self, state):
        if not self.__valid_factors: self.__initFactors()

        if hasattr(state.cache, "dMdt"): return state.cache.dMdt
        dMdt = state.cache.dMdt = VectorField(self.system.mesh)

        # Get effective field
        H_tot = self.calculate_H_tot(state)

        # Basic term
        magneto.llge(self.__f1, self.__f2, state.M, H_tot, dMdt)

        # Optional other terms
        for dMdt_id in self.llge_terms:
            dMdt_i = getattr(state, dMdt_id)
            dMdt.add(dMdt_i)

        return dMdt

    def calculate_deg_per_ns(self, state):
        if hasattr(state.cache, "deg_per_ns"): return state.cache.deg_per_ns
        deg_per_timestep = (180.0 / math.pi) * math.atan2(state.dMdt.absMax() * state.h, state.M.absMax()) # we assume a<b at atan(a/b).
        deg_per_ns = state.cache.deg_per_ns = 1e-9 * deg_per_timestep / state.h
        return deg_per_ns

    def __initFactors(self):
        self.__f1 = f1 = Field(self.system.mesh) # precession factors of llge
        self.__f2 = f2 = Field(self.system.mesh) # damping factors of llge

        alpha, Ms = self.alpha, self.Ms

        # Prepare factors
        for x,y,z in self.system.mesh.iterateCellIndices():
            alpha_i, Ms_i = alpha.get(x,y,z), Ms.get(x,y,z)

            if Ms_i != 0.0:
                gamma_prime = GYROMAGNETIC_RATIO / (1.0 + alpha_i**2)
                f1_i = -gamma_prime
                f2_i = -alpha_i * gamma_prime / Ms_i
            else:
                f1_i, f2_i = 0.0, 0.0

            f1.set(x, y, z, f1_i)
            f2.set(x, y, z, f2_i)

        # If precession is disabled, blank f1.
        if not self.__do_precess:
            f1.fill(0.0)

        # Done.
        self.__valid_factors = True
