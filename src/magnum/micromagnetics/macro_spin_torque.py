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

import magnum.module as module
import magnum.magneto as magneto
from magnum.mesh import VectorField, Field

# void fdm_slonchewski(
# 	int dim_x, int dim_y, int dim_z,
# 	double delta_x, double delta_y, double delta_z,
# 	double a_j,
# 	const VectorMatrix &p, // spin polarization
# 	const Matrix &Ms,
# 	const Matrix &alpha,
# 	const VectorMatrix &M,
# 	VectorMatrix &dM
# );

class MacroSpinTorque(module.Module):
  def __init__(self, do_precess = True):
    super(SpinTorque, self).__init__()
    self.__do_precess = do_precess
    raise NotImplementedError("The MacroSpinTorque module does not work yet.")
      
  def calculates(self):
    return ["dMdt_ST"]

  def params(self):
    return ["a_j", "p"]

  def properties(self):
    return {'LLGE_TERM': "dMdt_ST"}

  def initialize(self, system):
    self.system = system
    self.a_j = Field(self.system.mesh); self.a_j.fill(0.0)
    self.p = Field(self.system.mesh); self.p.fill(0.0)

  def calculate(self, state, id):
    cache = state.cache

    if id == "dMdt_ST":
      if hasattr(cache, "dMdt_ST"): return cache.dMdt_ST
      dMdt_ST = cache.dMdt_ST = VectorField(self.system.mesh)

      # Calculate macro spin torque term due to Slonchewski
      nx, ny, nz = self.system.mesh.num_nodes
      dx, dy, dz = self.system.mesh.delta
      magneto.fdm_slonchewski(
        nx, ny, nz, dx, dy, dz, #self.__do_precess,
        a_j, p, state.Ms, state.alpha,
        state.M, dMdt_ST
      )
      return dMdt_ST

    else:
      raise KeyError("MacroSpinTorque.calculate: Can't calculate %s", id)
