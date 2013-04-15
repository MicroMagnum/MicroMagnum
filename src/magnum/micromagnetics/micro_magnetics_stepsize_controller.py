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

from __future__ import print_function
from magnum.evolver import StepSizeController
import magnum.magneto as magneto

class MicroMagneticsStepSizeController(StepSizeController):
    def __init__(self, eps_abs = 1e-4, eps_rel = 1e-4):
        super(MicroMagneticsStepSizeController, self).__init__()

        self.allowed_absolute_error = eps_abs
        self.allowed_relative_error = eps_rel

        self.STEP_HEADROOM = 0.85
        self.MIN_TIMESTEP_SCALE = 0.2
        self.MAX_TIMESTEP_SCALE = 5.0
        self.MIN_TIMESTEP = 1e-15
        self.MAX_TIMESTEP = 1e-10

    def adjust_stepsize(self, state, h, order, y, y_err, dydt):
        # [y] = A/m.
        # [dydt] = A/m (deriviative of y).
        # [y_err] = A/m (estimated error dydt).
        # - One RK step: y = y_prev + (h*dydt +- y_err)

        max_y_err = magneto.scaled_abs_max(y_err, state.Ms) # max_y_err = max_i(|y_err[i]| / Ms[i]), [max_y_err] = rad
        max_dydt  = magneto.scaled_abs_max(dydt, state.Ms)  # max_dydt  = max_i(|dydt[i]| / Ms[i]),  [max_dydt]  = rad/s

        # Calculate estimated absolute and relative error.
        try:
            e_abs = max_y_err            # [e_abs] = rad
            e_rel = e_abs / (h*max_dydt) # [e_rel] = 1
        except ZeroDivisionError:
            print("ZeroDivisionError!")
            return True, 5e-14

        # Accept the step only with tolerable errors.
        accept = (e_abs <= self.allowed_absolute_error) and (e_rel <= self.allowed_relative_error)

        # Determine step size scaling factor for allowed absolute and relative errors.
        h_scale_abs = self.STEP_HEADROOM * pow(self.allowed_absolute_error / e_abs, 1.0/(order+1.0))
        h_scale_rel = self.STEP_HEADROOM * pow(self.allowed_relative_error / e_rel, 1.0/order)

        def clamp(x, min_x, max_x):
            if x < min_x: x = min_x
            if x > max_x: x = max_x
            return x

        # Take the most pessimistic (= smallest) scaling factor and clamp by [max_step_decrease, max_step_increase]
        h_scale = clamp(min([h_scale_abs, h_scale_rel]), self.MIN_TIMESTEP_SCALE, self.MAX_TIMESTEP_SCALE)

        # Determine next step size and clamp by [min_h, max_h]
        h_next = clamp(h * h_scale, self.MIN_TIMESTEP, self.MAX_TIMESTEP)

        # Accept step anyway if time step can not go any smaller.
        if h_next == self.MIN_TIMESTEP: accept = True

        return accept, h_next

    def __str__(self):
        return "MMM(eps_abs=%s, eps_rel=%s)" % (self.allowed_absolute_error, self.allowed_relative_error)
