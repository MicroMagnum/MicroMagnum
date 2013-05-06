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
from magnum.mesh import VectorField
from magnum.logger import logger

class SimpleVectorField(module.Module):
    def __init__(self, var_id):
        super(SimpleVectorField, self).__init__()
        self.__var_id = var_id

    def params(self):
        return [self.__var_id]

    def initialize(self, system):
        logger.info("%s: Providing parameters %s" % (self.name(), ", ".join(self.params())))

        A = VectorField(system.mesh)
        A.clear()

        setattr(self, self.__var_id, A)

# This module is for use as an external field term in the LLG equation
class SimpleExternalField(SimpleVectorField):
    def __init__(self, var_id):
        super(SimpleExternalField, self).__init__(var_id)
        self.__var_id = var_id

    def properties(self):
        return {'EFFECTIVE_FIELD_TERM': self.__var_id}
