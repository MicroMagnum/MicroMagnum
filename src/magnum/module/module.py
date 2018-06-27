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

from magnum.module.assign import assign

class Module(object):

    # These methods return information about the module:

    def calculates(self):
        return []

    def updates(self):
        return []

    def params(self):
        return []

    def properties(self):
        return {}

    def name(self):
        return self.__class__.__name__

    # Default implementation to set/get parameters via setattr/getattr on the module instance:

    def set_param(self, id, value, mask=None):
        p = getattr(self, id)
        p = assign(p, value, mask)
        setattr(self, id, p)

    def get_param(self, id):
        return getattr(self, id)

    def on_param_update(self, id):
        pass

    # Methods that have to be implemented:

    def initialize(self, system):
        raise NotImplementedError("Module.initialize")

    def calculate(self, state, id):
        raise NotImplementedError("Module.calculate")

    def update(self, state, id, val):
        raise NotImplementedError("Module.update")
