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

from .alternating_field import AlternatingField
from .constants import MU0

class ExternalField(AlternatingField):
  def __init__(self):
    super(ExternalField, self).__init__("H_ext")

  def calculates(self):
    return super(ExternalField, self).calculates() + ["E_ext"]

  def properties(self):
    p = super(ExternalField, self).properties()
    p.update({'EFFECTIVE_FIELD_TERM': "H_ext", 'EFFECTIVE_FIELD_ENERGY': "E_ext"})
    return p

  def calculate(self, state, id):
    if id == "E_ext":
      return -MU0 * self.system.mesh.cell_volume * state.M.dotSum(state.H_ext)
    else:
      return super(ExternalField, self).calculate(state, id)
