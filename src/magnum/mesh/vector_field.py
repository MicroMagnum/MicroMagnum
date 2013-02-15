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

class VectorField(magneto.VectorMatrix):
  def __init__(self, mesh, id = None, value_unit = None):
    super(VectorField, self).__init__(magneto.Shape(*mesh.getFieldMatrixDimensions()))
    self.__mesh = mesh
    self.__id = id
    self.__value_unit = value_unit

  def __repr__(self):
    return "VectorField(%r, id=%r, value_unit=%r)" % (self.__mesh, self.__id, self.__value_unit)

  def getMesh(self): return self.__mesh
  mesh = property(getMesh)

  def getValueUnit(self): return self.__value_unit
  value_unit = property(getValueUnit)

  def initFromFunction(self, init_fn):
    mesh = self.mesh
    for idx in range(mesh.total_nodes):
      self.set(idx, init_fn(mesh, mesh.getPosition(idx)))

  def findExtremum(self, z_slice=0, component=0): # TODO: Better name (only the xy-Plane is searched)
    if not isinstance(self.mesh, RectangularMesh):
      raise TypeError("VectorField.findExtremum: Vector fields on non-rectangular meshes are not supported!")
    cell = magneto.findExtremum(self, z_slice, component)
    return (
      (0.5 + cell[0]) * self.mesh.delta[0],
      (0.5 + cell[1]) * self.mesh.delta[1],
      (0.5 + cell[2]) * self.mesh.delta[2]
    )

  def interpolate(self, mesh):
    if not isinstance(mesh, RectangularMesh):
      raise TypeError("VectorField.interpolate: Fixme: Non-rectangular meshes are not supported!")

    # Get matrix with interpolated values in 'interp_mat'   
    need_interpolate = (self.mesh.num_nodes != mesh.num_nodes)
    if need_interpolate:
      nx, ny, nz = mesh.num_nodes # new size (in number of cells)
      interp_mat = magneto.linearInterpolate(self, magneto.Shape(nx, ny, nz))
    else:
      interp_mat = self # no need to interpolate..

    # Create interpolated vector field from matrix 'interp_mat'
    result = VectorField(mesh)
    result.assign(interp_mat)
    return result
