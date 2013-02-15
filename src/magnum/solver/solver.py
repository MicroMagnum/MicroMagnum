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

import magnum.module as module
import magnum.evolver as evolver
import magnum.tools as tools

from .step_handler import StepHandler
from .condition import Condition

class Solver(object):
  class FinishSolving(Exception): pass
  class StartDebugger(Exception): pass

  def __init__(self, system, _evolver):
    if not isinstance(system, module.System): raise TypeError("The 'system' argument must be a module.System instance.")
    if not isinstance(_evolver, evolver.Evolver): raise TypeError("The 'evolver' argument must be a evolver.Evolver instance.")

    self.__system = system
    self.__state = system.createState()
    self.__evolver = _evolver
    self.__step_handlers = []

  def __repr__(self):
    return "Solver@%s" % hex(id(self))

  def __del__(self):
    # notify the step handlers.
    for step_handler, _ in self.__step_handlers: step_handler.done()

  @property
  def state(self): return self.__state

  @property
  def step_handlers(self): return self.__step_handlers[:]

  @property
  def model(self): return self.__system

  def addStepHandler(self, step_handler, condition):
    if not isinstance(step_handler, StepHandler): raise TypeError("'step_handler' argument must be a StepHandler")
    if not isinstance(condition, Condition): raise TypeError("'condition' argument must be a Condition")
    if any(s == step_handler for s, c in self.__step_handlers): raise ValueError("Cannot add step handler more than once")
    self.__step_handlers.append((step_handler, condition))

  ### The solver loop ##############################################################

  def step_with_t_max(self, t_max):
    # I. Get "smallest time of interest in the future" from the step handlers
    t0 = self.__state.t
    t1 = min(filter(lambda t: t is not None and t > t0, [t_max] + [c.get_time_of_interest(self.__state) for s, c in self.__step_handlers]))

    # II. Do step from t0 to up to t1.
    self.__state = self.__evolver.evolve(self.__state, t1)

    # III. Call step handlers
    self.__call_step_handlers()   

  def step(self):
    self.step_with_t_max(1e100)

  def solve(self, stop_condition):
    self.state.flush_cache()
    tools.flush()

    # Run solver loop.
    # Also, custom sigint handler while loop is running.
    import signal
    self.__interrupted = False
    def my_int_handler(signum, frame):
      assert signum == signal.SIGINT
      self.__interrupted = True
    old_sigint_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, my_int_handler) # install sigint handler
    try:
      self.__solve_loop(stop_condition) # run solver loop
    finally:
      signal.signal(signal.SIGINT, old_sigint_handler) # uninstall sigint handler
      del self.__interrupted

    self.state.flush_cache()
    tools.flush()

  def __call_step_handlers(self):
    for step_handler, condition in self.__step_handlers:
      if condition.check(self.__state):
        step_handler.handle(self.__state)

  def __solve_loop(self, stop_condition):
    self.__call_step_handlers() # Because this makes sense.

    while not stop_condition.check(self.__state):
      # Evolve..
      self.step_with_t_max(stop_condition.get_time_of_interest(self.__state) or 1e100)
      
      # Handle Ctrl-C interruption
      if self.__interrupted:
        do_finish = False
        try:
          self.handle_interrupt()
        except Solver.FinishSolving:
          do_finish = True
        except Solver.StartDebugger:
          print("Entering Python debugger. Type 'c' to exit debugging.")
          print("The currect state can be accessed with the expression 'self.state'.")
          print()
          import pdb; pdb.set_trace()
        finally:
          self.__interrupted = False
        if do_finish: break # pretend that stop_condition is true.

  def handle_interrupt(self):
    raise KeyboardInterrupt()
