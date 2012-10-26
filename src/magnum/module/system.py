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

from magnum.logger import logger

import magnum.evolver as evolver

class System(object):
  def __init__(self, mesh):
    self.mesh = mesh
    self.vars = {}
    self.calculators = {}
    self.updaters = {}
    self.param_handlers = {}
    self.modules = []
    self.initialized = False

  def addModule(self, module):
    self.modules.append(module)

  def createState(self):
    if not self.initialized: self.initialize()

    # Create class
    class State(evolver.State):
      def __init__(this):
        super(State, this).__init__(self.mesh)
    self.imbue_state_class_properties(State)
    return State()

  def calculate(self, state, id):
    mod = self.calculators[id]
    return mod.calculate(state, id)

  def update(self, state, id, value):
    mod = self.updaters[id]
    return mod.update(state, id, value)

  def get_param(self, id):
    mod = self.param_handlers[id]
    return mod.get_param(id)

  def set_param(self, id, value, mask = None):
    mod = self.param_handlers[id]
    mod.set_param(id, value, mask)
    # notify modules
    for mod in self.modules: mod.on_param_update(id)

  def canUpdate(self, id):
    return id in self.updaters.keys()

  def canCalculate(self, id):
    return id in self.calculators.keys()

  def canHandleParam(self, id):
    return id in self.param_handlers.keys()

  all_params = property(lambda self: list(self.param_handlers.keys()))

  def initialize(self):
    names = []
    for mod in self.modules: 
      for var in mod.calculates():
        if var in self.calculators: raise ValueError("There exist at least two modules that want to calculate variable '%s': Modules %s and %s." % (var, mod.__class__.__name__, self.calculators[var].__class__.__name__))
        self.calculators[var] = mod
      for var in mod.updates(): 
        if var in self.updaters: raise ValueError("There exist at least two modules that want to update/change variable '%s': Modules %s and %s." % (var, mod.__class__.__name__, self.updaters[var].__class__.__name__))
        self.updaters[var] = mod
      for var in mod.params(): 
        if var in self.updaters: raise ValueError("There exist at least two modules that want to handle parameter '%s': Modules %s and %s." % (var, mod.__class__.__name__, self.param_handlers[var].__class__.__name__))
        self.param_handlers[var] = mod
      mod.initialize(self)
      names.append(type(mod).__name__)
    logger.info("Initialized modules: %s", ", ".join(names))
    self.initialized = True

  # Model Parameters #

  def __getattr__(self, key):
    if key != 'param_handlers' and key in self.__dict__['param_handlers'].keys():
      return self.get_param(key)
    else:
      return object.__getattribute__(self, key)

  def __setattr__(self, key, value):
    if hasattr(self, 'param_handlers') and key in self.__dict__['param_handlers'].keys():
      self.set_param(key, value)
    else:
      super(System, self).__setattr__(key, value)

  # Add accessors for model variables to the state class

  def imbue_state_class_properties(self, state_class):
    assert self.initialized
    assert issubclass(state_class, evolver.State)

    def make_param_property(var): 
      return property(
        lambda state       : self.get_param(var),
        lambda state, value: self.set_param(var, value)
      )

    def make_var_property(var):
      calc_mod = self.calculators.get(var, None)
      if calc_mod:
        getter = lambda state: calc_mod.calculate(state, var)
      else:
        def getter(state): raise KeyError("Model variable %s is not readable." % var)
      upd_mod = self.updaters.get(var, None)
      if upd_mod:
        setter = lambda state, val: upd_mod.update(state, var, val)
      else:
        def setter(state, val): raise KeyError("Model variable %s is not writable." % var)
      return property(getter, setter)
      
    # I. Properties for model parameters
    for var, mod in self.param_handlers.items(): setattr(state_class, var, make_param_property(var))

    # II. Properties for model variables
    vars = list(self.calculators.keys()) + list(self.updaters.keys())
    for var in vars: setattr(state_class, var, make_var_property(var))

    # III. "system" property resolving to self.
    state_class.system = property(lambda state: self)
