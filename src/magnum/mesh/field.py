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

import magnum.magneto as magneto

from .rectangular_mesh import RectangularMesh

class Field(magneto.Matrix):
  def __init__(self, mesh):
    nx, ny, nz = mesh.num_nodes
    magneto.Matrix.__init__(self, magneto.Shape(nx, ny, nz))
    self.__mesh = mesh

  def __repr__(self):
    return "Field(%r)" % self.__mesh

  def getMesh(self): return self.__mesh
  mesh = property(getMesh)

  def interpolate(self, mesh):
    if not isinstance(mesh, RectangularMesh):
      raise TypeError("Field.interpolate: Fixme: Non-rectangular meshes are not supported!")

    # Get matrix with interpolated values in 'interp_mat'   
    need_interpolate = (self.mesh.num_nodes != mesh.num_nodes)
    if need_interpolate:
      nx, ny, nz = mesh.num_nodes # new size (in number of cells)
      interp_mat = magneto.linearInterpolate(self, magneto.Shape(nx, ny, nz))
    else:
      interp_mat = self # no need to interpolate..

    # Create interpolated vector field from matrix 'interp_mat'
    result = Field(mesh)
    result.assign(interp_mat)
    return result
