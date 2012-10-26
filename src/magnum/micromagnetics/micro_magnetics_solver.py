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

from __future__ import print_function

import magnum.console as console 
import magnum.solver as solver
import magnum.evolver as evolver

from .micro_magnetics import MicroMagnetics 
from .io import writeOMF

class MicroMagneticsSolver(solver.Solver):
  def __init__(self, modsys, evolver, world):
    super(MicroMagneticsSolver, self).__init__(modsys, evolver)
    self.__world = world

  def __repr__(self):
    return "MicroMagneticsSolver@%s" % hex(id(self))

  world = property(lambda self: self.__world)

  def relax(self, *args, **kwargs):
    return self.solve(solver.condition.Relaxed(*args, **kwargs))

  def handle_interrupt(self):
    print()
    print(self.state)
    answer = console.interactive_menu(
      header = "Solver interrupted by signal SIGINT.",
      text = "Your options:",
      options = [
        "Continue solving.",
        "Stop solver and return the current state as the result.",
        "Save current magnetization to .omf file, then continue.",
        "Raise KeyboardInterrupt (graceful program exit)",
        "Kill program"
      ]
    )
    if answer == 1:
      return
    elif answer == 2:
      raise solver.Solver.FinishSolving
    elif answer == 3:
      print("Enter file name ('.omf' is appended automatically)")
      path = console.getline() + ".omf"
      writeOMF(path, self.state.M)
      print("Done. Continuing..")
      return False
    elif answer == 4:
      raise KeyboardInterrupt
    elif answer == 5:
      import sys
      sys.exit(-1)
    assert False
