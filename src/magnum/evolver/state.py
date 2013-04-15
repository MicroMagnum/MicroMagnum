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

from copy import copy

class State(object):
    class Cache(object):
        pass

    def __init__(self, mesh):
        self.t = 0
        self.h = 0
        self.step = 0
        self.substep = 0
        self.mesh = mesh
        self.y = VectorField(mesh)
        self.flush_cache()

    def differentiate(self, dst):
        raise NotImplementedError("State.differentiate")

    cache = property(lambda self: self.__cache)

    def flush_cache(self):
        self.__cache = State.Cache()

    def finish_step(self):
        pass

    def clone(self, y_replacement):
        state = copy(self)
        state.y = y_replacement
        state.flush_cache()
        return state
