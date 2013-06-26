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

from .micro_magnetics import MicroMagnetics
from .micro_magnetics_solver import MicroMagneticsSolver
from .micro_magnetics_stepsize_controller import MicroMagneticsStepSizeController
from .landau_lifshitz_gilbert import LandauLifshitzGilbert
from .stephandler import ScreenLog

import magnum.evolver as evolver
import magnum.solver.condition as condition

def create_solver(world, module_list = [], **kwargs):

    ###### I. Create module system ###############################
    sys = MicroMagnetics(world)
    sys.addModule(LandauLifshitzGilbert(do_precess = kwargs.pop("do_precess", True)))
    for mod in module_list:
        if isinstance(mod, type):
            inst = mod()
        else:
            inst = mod
        sys.addModule(inst)
    sys.initialize()
    sys.initializeFromWorld()

    ###### II. Create Evolver ####################################
    evolver_id = kwargs.pop("evolver", "rkf45")
    if evolver_id in ["rkf45", "rk23", "cc45", "dp54"]:
        eps_rel = kwargs.pop("eps_rel", 1e-4)
        eps_abs = kwargs.pop("eps_abs", 1e-3)
        evo = evolver.RungeKutta(sys.mesh, evolver_id, MicroMagneticsStepSizeController(eps_abs, eps_rel))
    elif evolver_id in ["rkf45x", "rk23x", "cc45x", "dp54x"]:
        eps_rel = kwargs.pop("eps_rel", 1e-4)
        eps_abs = kwargs.pop("eps_abs", 1e-3)
        evo = evolver.RungeKutta(sys.mesh, evolver_id, evolver.NRStepSizeController(eps_abs, eps_rel))
    elif evolver_id in ["rk4"]:
        step_size = kwargs.pop("step_size", 3e-13)
        evo = evolver.RungeKutta4(sys.mesh, step_size)
    elif evolver_id in ["euler"]:
        step_size = kwargs.pop("step_size", 5e-15)
        evo = evolver.Euler(sys.mesh, step_size)
    elif evolver_id in ["cvode"]:
        eps_rel = kwargs.pop("eps_rel", 1e-4)
        eps_abs = kwargs.pop("eps_abs", 1e-3)
        newton_method = kwargs.pop("newton_method", False)
        step_size = kwargs.pop("step_size", 1e-12)
        evo = evolver.Cvode(sys.mesh, eps_abs, eps_rel, step_size, newton_method)
    else:
        raise ValueError("Invalid evolver type specified: %s (valid choices: 'rk23','rkf45','dp54','euler','cvode'; default is 'rkf45')" % evolver_id)

    ###### III. Create Solver from Evolver and module system ######
    solver = MicroMagneticsSolver(sys, evo, world)

    ###### IV. Optional: Add screen log ###########################
    log_enabled = kwargs.pop("log", False)
    if log_enabled:
        n = log_enabled if type(log_enabled) == int else 100
        solver.addStepHandler(ScreenLog(), condition.EveryNthStep(n))

    # Check if any arguments could not be parsed.
    if len(kwargs) > 0:
        raise ValueError("create_solver: Excess parameters given: %s" % ", ".join(kwargs.keys()))

    return solver
