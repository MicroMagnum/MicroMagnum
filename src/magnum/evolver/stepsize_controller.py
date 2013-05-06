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

import magnum.magneto as magneto

class StepSizeController(object):
    def __init__(self):
        pass

    def adjust_stepsize(self, state, h, order, y, dydt, y_err):
        raise NotImplementedError("StepSizeController.adjust_stepsize")
        #return accept, new_h

# Algorithm from Numerical Recepies (NR) book
class NRStepSizeController(StepSizeController):
    def __init__(self, eps_abs = 1e-3, eps_rel = 1e-3):
        self.eps_abs = eps_abs
        self.eps_rel = eps_rel

    def adjust_stepsize(self, state, h, order, y, y_err, dydt):
        h_new = magneto.rk_adjust_stepsize(order, h, self.eps_abs, self.eps_rel, y, y_err)
        accept = (h_new <= h)
        return accept, h_new

    def __str__(self):
        return "NR(eps_abs=%s, eps_rel=%s)" % (self.eps_abs, self.eps_rel)

class FixedStepSizeController(StepSizeController):
    def __init__(self, h):
        self.h = h

    def adjust_stepsize(self, state, h, order, y, y_err, dydt):
        return True, self.h
