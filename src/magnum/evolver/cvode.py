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
from magnum.llgDiffEq import *

import magnum.magneto as m

class Cvode(Evolver):
  def __init__(self, mesh, eps_abs, eps_rel, step_size):
    super(Cvode, self).__init__(mesh)
    self.eps_abs = eps_abs
    self.eps_rel = eps_rel
    self.step_size = step_size

  def initialize(self, state):
    self.llg = LlgDiffEq(state)
    self.cvode = m.Cvode(self.llg, self.eps_abs, self.eps_rel)
    state.h = self.step_size

  def evolve(self, state, t_max):
    if not hasattr(Cvode, 'cvode'):
      self.initialize(state)

    #state.y.add(dydt, self.step_size)

    # call cvode
    self.cvode.evolve(state.t, state.h)

    #state.y.add(dydt, self.step_size)
    state.t += state.h
    state.step += 1
    state.substep = 0
    state.flush_cache()
    state.finish_step()
    return state
