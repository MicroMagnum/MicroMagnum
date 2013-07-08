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

# evolvers
from .evolver import Evolver # base class
from .euler import Euler
from .runge_kutta import RungeKutta
from .runge_kutta_4 import RungeKutta4
from .stepsize_controller import StepSizeController, NRStepSizeController, FixedStepSizeController

try:
  from .cvode import Cvode
  have_cvode = True
except ImportError:
  have_cvode = False
except AttributeError: # cvode raises an AttributeError.
  have_cvode = False


# evolver state class
from .state import State

__all__ = ["Evolver", "Euler", "RungeKutta", "RungeKutta4", "StepSizeController", "NRStepSizeController", "FixedStepSizeController", "State"]

if have_cvode:
  __all__ += ["Cvode"]
