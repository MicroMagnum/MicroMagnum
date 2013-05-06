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
import magnum.evolver as evolver
from magnum.mesh import Field, VectorField
from magnum.logger import logger

import numbers

class MicroMagnetics(module.System):
    def __init__(self, world):
        super(MicroMagnetics, self).__init__(world.mesh)

        self.__state_class = None
        self.__world = world

    world = property(lambda self: self.__world)

    def createState(self):
        state_class = self.__make_state_class()

        state = state_class()
        state.h = 1e-13
        return state

    def __make_state_class(self):
        if not self.__state_class:
            if not self.initialized: self.initialize()

            class MicroMagneticsState(evolver.State):
                def __init__(this):
                    super(MicroMagneticsState, this).__init__(self.mesh)

                def finish_step(this):
                    this.M.normalize(this.system.Ms)

                def differentiate(this):
                    return this.dMdt

                def __getitem__(this, key):
                    if self.world.hasBody(key):
                        return BodyProxy(self, this, self.world.findBody(key)) # BodyProxy defined at bottom of this file.
                    else:
                        raise KeyError("No such body: %s" % key)
            # end class

            self.imbue_state_class_properties(MicroMagneticsState)
            self.__state_class = MicroMagneticsState

        return self.__state_class

    def initializeFromWorld(self):
        logger.info("Initializing material parameters")

        def format_parameter_value(value):
            if isinstance(value, numbers.Number):
                if abs(value) >= 1e3 or abs(value) <= 1e-3: return "%g" % value
            return str(value)

        for body in self.world.bodies:
            mat   = body.material
            cells = body.shape.getCellIndices(self.mesh)

            used_param_list = []
            for param in self.all_params:
                if hasattr(mat, param):
                    val = getattr(mat, param)
                    self.set_param(param, val, mask=cells)
                    used_param_list.append("'%s=%s'" % (param, format_parameter_value(val)))

            logger.info("  body id='%s', volume=%s%%, params: %s",
              body.id,
              round(1000.0 * len(cells) / self.mesh.total_nodes) / 10,
              ", ".join(used_param_list) or "(none)"
            )

class BodyProxy(object):
    def __init__(self, system, state, body):
        self.__state = state
        self.__mask = body.shape.getCellIndices(system.mesh)

    def __setattr__(self, key, val):
        try:
            state = self.__state
            mask  = self.__mask
        except:
            return super(BodyProxy, self).__setattr__(key, val)

        old = getattr(state, key)
        if not isinstance(old, (Field, VectorField)):
            raise KeyError("For a single body, can only assign to model variables that contain a field or vector field: %s" % key)
        module.assign(old, val, mask)
        setattr(state, key, old)
