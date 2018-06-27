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

import magnum.module as module
from magnum.mesh import VectorField, Field
from magnum.logger import logger


class CurrentPath(module.Module):

    def __init__(self):
        super(CurrentPath, self).__init__()

    def calculates(self):
        return [
            # Current density (A/m^2)                
            "j",

            # Electric potential (todo)
            "phi"  
        ]

    def params(self):
        return [
            # Applied voltage
            "U"
        ]

    def properties(self):
        return {}

    def initialize(self, system):
        self.system = system

        self.U = 2.0
        logger.info("%s: Providing model variables %s, parameters are %s" % (self.name(), ", ".join(self.calculates()), ", ".join(self.params())))

    def on_param_update(self, id):
        if id == "U":
            logger.info("Received new voltage: U = {self.U}V".format(self=self))

    def calculate(self, state, id):
        if id == "phi":
            phi = Field(self.system.mesh)
            phi.fill(0.0)
            return phi

        elif id == "j":
            J = VectorField(self.system.mesh)
            J.fill([0.0, 0.0, 0.0])
            return J

        else:
            raise KeyError("AlternatingField.calculates: Can't calculate %s", id)

