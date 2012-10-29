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

import itertools

class Mesh(object):
  pass

class RectangularMesh(Mesh):
  def __init__(self, num_nodes, cell_size, periodic_bc="", periodic_repeat=None):
    assert isinstance(num_nodes, tuple) and len(num_nodes) == 3
    assert isinstance(cell_size, tuple) and len(cell_size) == 3
    assert all([isinstance(x, (int, float)) for x in num_nodes])
    cell_size = tuple(map(float,cell_size))
    num_nodes = tuple(map(int,num_nodes)) # XXX: Warn on round down???

    nx, ny, nz = num_nodes
    dx, dy, dz = cell_size

    self.__volume = nx*ny*nz * dx*dy*dz
    self.__num_nodes = num_nodes
    self.__delta = cell_size
    self.__periodic_bc = periodic_bc

    if periodic_repeat is None:
      # count number of periodic directions (0, 1, 2, or 3)
      num_dirs = 0
      for s in ["x", "y", "z"]: 
        if periodic_bc.find(s) != -1: num_dirs += 1
      # defaults for peri_repeat for counted num_dirs.
      periodic_repeat = {0:1, 1:15, 2:3, 3:2}[num_dirs]
    self.__periodic_repeat = periodic_repeat

  def __repr__(self):
    return "RectangularMesh(%r, %r, %r)" % (
      self.num_nodes,
      self.delta,
      self.__periodic_bc
    )

  def isCompatible(self, other): 
    # XXX: What about periodic boundary conditions?
    return self.num_nodes == other.num_nodes and self.delta == other.delta

  def getFieldMatrixDimensions(self):
    return self.__num_nodes

  def getVolume(self):
    """
    Returns the volume of the complete mesh.
    """
    return self.__volume

  def getCellVolume(self):
    """
    Returns the volume of one cell.
    """
    return self.__delta[0] * self.__delta[1] * self.__delta[2]

  def getTotalNodes(self):
    """
    Returns the total number of cells.
    """
    return self.__num_nodes[0] * self.__num_nodes[1] * self.__num_nodes[2]

  def getNumNodes(self):
    """
    Returns the number of cells in x-,y-, and z-direction as a 3-tuple.
    """
    return self.__num_nodes

  def getDelta(self):
    """
    Returns the cell dimensions as a 3-tuple.
    """
    return self.__delta

  def getSize(self):
    """
    Returns the size of the mesh in x-,y- and z-direction as a 3-tuple.
    """
    nx, ny, nz = self.num_nodes
    dx, dy, dz = self.delta
    return nx*dx, ny*dy, nz*dz

  def getPeriodicBC(self):
    return self.__periodic_bc, self.__periodic_repeat

  def getPosition(self, linidx):
    """
    Returns the middle point of the cell with linear index 'linidx'.
    """
    nx, ny, nz = self.num_nodes
    dx, dy, dz = self.delta

    # Get x,y,z index from linear index.
    stride_x = 1
    stride_y = nx * stride_x
    stride_z = ny * stride_y
    
    z = (linidx           ) // stride_z
    y = (linidx % stride_z) // stride_y
    x = (linidx % stride_y) // stride_x

    # return pos (of cell center, hence +0.5)
    return (x+0.5)*dx, (y+0.5)*dy, (z+0.5)*dz

  def iterateCellIndices(self):
    # Returns iterator that iterates through all cell indices (x,y,z).
    # Ex.:
    #   for x,y,z in mesh.iterateCellIndices():
    #     print(x,y,z)
    return itertools.product(*map(range, self.num_nodes))
  
  volume = property(getVolume)
  cell_volume = property(getCellVolume)
  total_nodes = property(getTotalNodes)
  num_nodes = property(getNumNodes)
  delta = property(getDelta)
  size = property(getSize)
  periodic_bc = property(getPeriodicBC)
