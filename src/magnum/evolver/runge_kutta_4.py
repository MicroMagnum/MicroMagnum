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

from .evolver import Evolver

class RungeKutta4(Evolver):
    def __init__(self, mesh, step_size):
        super(RungeKutta4, self).__init__(mesh)
        self.step_size = float(step_size)

    # TODO: We waste memory in this function because 4 state objects exist at the same time!
    def evolve(self, state, t_max):
        h = self.step_size

        s0 = state
        k0 = s0.differentiate(); s0.flush_cache()

        s1 = s0.clone(); s1.substep = 1; s1.t = s0.t + h/2
        s1.y.assign(state.y)
        s1.y.add(k0, h/2)
        k1 = s1.differentiate(); s1.flush_cache()

        s2 = s0.clone(); s2.substep = 2; s2.t = s0.t + h/2
        s2.y.assign(state.y)
        s2.y.add(k1, h/2)
        k2 = s2.differentiate(); s2.flush_cache()

        s3 = s0.clone(); s3.substep = 3; s3.t = s0.t + h
        s3.y.assign(state.y)
        s3.y.add(k2, h/2)
        k3 = s3.differentiate(); s3.flush_cache()

        state.y.add(k0, h * 1.0/6.0)
        state.y.add(k1, h * 2.0/6.0)
        state.y.add(k2, h * 2.0/6.0)
        state.y.add(k3, h * 1.0/6.0)

        state.t += h
        state.h = h
        state.step += 1
        state.substep = 0
        state.finish_step()
        return state
