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
from magnum.logger import logger

from magnum.magneto import rk_prepare_step, rk_combine_result

from magnum.evolver.evolver import Evolver
from magnum.evolver.tableaus import rkf45, cc45, dp54, rk23


class RungeKutta(Evolver):
    TABLES = {
        'rkf45': rkf45,  # Runge-Kutta-Fehlberg
        'cc45': cc45,    # Cash-Karp
        'dp54': dp54,    # Dormand-Prince
        'rk23': rk23,    # Bogacki-Shampine
    }

    def __init__(self, mesh, method, stepsize_controller):
        super(RungeKutta, self).__init__(mesh)

        self.tab = RungeKutta.TABLES[method]()
        self.controller = stepsize_controller

        self.y0 = VectorField(mesh)
        self.y_err = VectorField(mesh)
        self.y_tmp = VectorField(mesh)
        self.k = [None] * self.tab.getNumSteps()

        logger.info("Runge Kutta evolver: method is %s, step size controller is %s.", method, self.controller)

    def evolve(self, state, t_max):
        # shortcuts
        y, y0, y_err = state.y, self.y0, self.y_err

        # This is needed to be able to roll back one step
        y0.assign(y)

        # Get time step to try.
        try:
            h_try = state.__runge_kutta_next_h
        except AttributeError:
            h_try = state.h

        while True:
            # Try a step from state.t to state.t+h_try
            dydt = self.apply(state, h_try)

            # Step size control (was h too big?)
            # calculate the minimal acceptable step size
            accept, h_new = self.controller.adjust_stepsize(state, h_try, self.tab.order, y, y_err, dydt)
            if accept:
                # done -> exit loop
                break
            else:
                # oh, tried step size was too large.
                y.assign(y0)  # reverse last step
                h_try = h_new  # try again with new (smaller) h.
                continue  # need to retry -> redo loop

        # But: Don't overshoot past t_max!
        if state.t + h_try > t_max:
            h_try = t_max - state.t   # make h_try smaller.
            y.assign(y0)              # reverse last step
            self.apply(state, h_try)  # assume that a smaller step size is o.k.

        # Update state
        state.t += h_try
        state.h = h_try
        state.__runge_kutta_next_h = h_new
        state.step += 1
        state.flush_cache()
        state.finish_step()
        return state

    def apply(self, state, h):
        tab = self.tab
        y, y_tmp, y_err, k = state.y, self.y_tmp, self.y_err, self.k
        num_steps = tab.getNumSteps()

        # I. Calculate step vectors k[0] to k[num_steps-1]

        state0 = state

        # step vector 0
        # (if method has first-step-as-last property, we might already
        #  know the first step vector.)
        k[0] = tab.fsal and getattr(state, "dydt_in", None) or state.differentiate()

        # step vectors 1 to (num_steps-1)
        for step in range(1, num_steps):
            # calculate ytmp...
            if num_steps != 6:
                # High-level version for num_steps != 6
                y_tmp.assign(y)
                for j in range(0, step):
                    y_tmp.add(k[j], h * tab.getB(step, j))
            else:
                # C++ version for num_steps==6 (rkf45,cc45)
                rk_prepare_step(step, h, tab, k[0], k[1] or k[0], k[2] or k[0], k[3] or k[0], k[4] or k[0], k[5] or k[0], y, y_tmp)

            state1 = state0.clone(y_tmp)
            state1.t = state0.t + h * tab.getA(step)
            k[step] = state1.differentiate()

        # II. Linear-combine step vectors, add them to y, calculate error y_err
        if num_steps == 6:
            rk_combine_result(h, tab, k[0], k[1], k[2], k[3], k[4], k[5], y, y_err)
        elif num_steps == 3:
            rk_combine_result(h, tab, k[0], k[1], k[2], y, y_err)
        else:
            y_err.clear()
            for step in range(0, num_steps):
                y.add(k[step], h * tab.getC(step))
                y_err.add(k[step], h * tab.getEC(step))

        # III. Exploit fsal property?
        if tab.fsal:
            state.dydt_in = k[num_steps - 1]  # save last dydt for next step

        # Also, return dydt (which is k[0])
        return k[0]
