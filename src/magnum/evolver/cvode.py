# Copyright 2012 by the Micromagnum authors.
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

from .evolver import Evolver

import magnum.magneto as m

class Cvode(Evolver):
  def __init__(self, mesh):
    super(Cvode, self).__init__(mesh)
    self.llg = LlgDiffEq(state)
    self.cvode = m.Cvode(llg)

  def evolve(self, state, t_max):
#    state.y.add(dydt, self.step_size)
#    state.t += self.step_size
#    state.h = self.step_size
#    state.substep = 0
#    state.flush_cache()
#    state.finish_step()
    self.cvode.evolve(t_max)
    state.step += 1
    return state
