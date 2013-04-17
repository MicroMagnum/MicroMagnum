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

class MicroMagnetics(module.System):

    def __init__(self, world):
        super(MicroMagnetics, self).__init__(world.mesh)
        self.__world = world

    world = property(lambda self: self.__world)

    def createStateClass(self):
        try:
            cls = self.__state_cls

        except AttributeError:
            class BodyProxy(object):
                def __init__(this, state, body):
                    this.__state = state
                    this.__mask = body.shape.getCellIndices(self.mesh)
            
                def __setattr__(this, key, val):
                    try:
                        state = this.__state
                        mask  = this.__mask
                    except:
                        return super(BodyProxy, this).__setattr__(key, val)
            
                    old = getattr(state, key)
                    if not isinstance(old, (Field, VectorField)):
                        raise KeyError("For a single body, can only assign to model variables that contain a field or vector field: %s" % key)
                    module.assign(old, val, mask)
                    setattr(state, key, old)

            class MicroMagneticsState(evolver.State):
                def __init__(this, mesh):
                    super(MicroMagneticsState, this).__init__(mesh)
                    this.h = 1e-13

                def finish_step(this):
                    this.M.normalize(this.Ms)

                def differentiate(this):
                    return this.dMdt

                def __getitem__(this, key):
                    if self.world.hasBody(key):
                        return BodyProxy(state=this, body=self.world.findBody(key)) # BodyProxy defined at bottom of this file.
                    else:
                        raise KeyError("No such body: %s" % key)

            self.addStateClassProperties(MicroMagneticsState)
            cls = self.__state_class = MicroMagneticsState

        return cls
