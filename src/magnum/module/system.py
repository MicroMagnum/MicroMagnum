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

import magnum.logger as logger

class System(object):

    def __init__(self, mesh):
        self.mesh = mesh
        self.modules = []  # list of modules added by addModule
        self.initialized = False

    parameters = property(lambda self: list(self.param_handlers.keys()))

    def addModule(self, module):
        assert not self.initialized
        self.modules.append(module)

    def createState(self):
        assert self.initialized
        return self.createStateClass()(self.mesh)

    def createStateClass(self):
        assert self.initialized
        class DefaultState(object):
            pass
        self.addStateClassProperties(DefaultState)

    # Interface to modules

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
        for mod in self.modules: mod.on_param_update(id)  # notify modules

    # Initialization

    def initialize(self):
        assert not self.initialized

        # Create model-variable lookup tables

        self.calculators    = {}  # map model var id to calculating module
        self.updaters       = {}  # map model var id to updating module
        self.param_handlers = {}  # map model parameter id to handling module

        for mod in self.modules:
            for var in mod.calculates():
                if var in self.calculators:
                    raise ValueError("Conflicting modules that calculate variable '%s': %s and %s." % (var, mod.name(), self.calculators[var].name()))
                self.calculators[var] = mod

            for var in mod.updates():
                if var in self.updaters:
                    raise ValueError("Conflicting modules that update variable '%s': %s and %s." % (var, mod.name(), self.calculators[var].name()))
                self.updaters[var] = mod

            for var in mod.params():
                if var in self.updaters:
                    raise ValueError("Conflicting modules that handle parameter '%s': %s and %s." % (var, mod.name(), self.calculators[var].name()))
                self.param_handlers[var] = mod

        # Create custom state class
        self.CustomState = self.createStateClass()
        self.initialized = True

        # Initialize modules
        for mod in self.modules:
            mod.initialize(self)
        logger.info("Initialized modules: %s", ", ".join(mod.name() for mod in self.modules))

    ### Create custom state class with properties for a given set of modules ###

    def addStateClassProperties(self, state_cls):

        def make_param_property(var):

            def getter(state):
                return self.get_param(var)

            def setter(state, value):
                self.set_param(var, value)
                state.flush_cache() # just to be sure

            return property(getter, setter)

        def make_var_property(var):

            calc_mod = self.calculators.get(var, None)
            if calc_mod:
                def getter(state):
                    return calc_mod.calculate(state, var)
            else:
                def getter(state):
                    raise KeyError("Model variable '%s' is not readable." % var)

            upd_mod = self.updaters.get(var, None)
            if upd_mod:
                def setter(state, value):
                    upd_mod.update(state, var, value)
            else:
                def setter(state, value):
                    raise KeyError("Model variable '%s' is not changeable." % var)

            return property(getter, setter)

        # I. Add to class properties for model parameters
        for var, mod in self.param_handlers.items():
          setattr(state_cls, var, make_param_property(var))

        # II. Add to class properties for model variables
        for var in list(self.calculators.keys()) + list(self.updaters.keys()):
            setattr(state_cls, var, make_var_property(var))

        # III. Add getter for system.
        state_cls.system = property(lambda state: self)
