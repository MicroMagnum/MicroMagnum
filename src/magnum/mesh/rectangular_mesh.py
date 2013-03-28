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

import itertools
import magnum.magneto as magneto

class RectangularMesh(magneto.RectangularMesh):
  def __init__(self, num_nodes, cell_size, periodic_bc="", periodic_repeat=None):
    assert isinstance(num_nodes, tuple) and len(num_nodes) == 3
    assert isinstance(cell_size, tuple) and len(cell_size) == 3
    assert all([isinstance(x, (int, float)) for x in num_nodes])

    nx, ny, nz = tuple(map(int,num_nodes))
    dx, dy, dz = tuple(map(float,cell_size))

    if periodic_repeat is None:
      # count number of periodic directions (0, 1, 2, or 3)
      num_dirs = 0
      for s in ["x", "y", "z"]: 
        if periodic_bc.find(s) != -1: num_dirs += 1
      # defaults for peri_repeat for counted num_dirs.
      periodic_repeat = {0:1, 1:15, 2:3, 3:2}[num_dirs]
 
    super(RectangularMesh, self).__init__(nx, ny, nz, dx, dy, dz, periodic_bc, periodic_repeat)

  def __repr__(self):
    return "RectangularMesh(%r, %r, periodic_bc=%r, periodic_repeat=%r)" % (self.num_nodes, self.delta, self.periodic_bc[0], self.periodic_bc[1])

  def getFieldMatrixDimensions(self):
    return self.__num_nodes

  def iterateCellIndices(self):
    """
    Returns iterator that iterates through all cell indices (x,y,z).
    Example:
      for x,y,z in mesh.iterateCellIndices():
        print(x,y,z)
    """
    return itertools.product(*map(range, self.num_nodes))
  
  volume = property(magneto.RectangularMesh.getVolume)
  cell_volume = property(magneto.RectangularMesh.getCellVolume)
  total_nodes = property(magneto.RectangularMesh.getTotalNodes)
  num_nodes = property(magneto.RectangularMesh.getNumNodes)
  delta = property(magneto.RectangularMesh.getDelta)
  size = property(magneto.RectangularMesh.getSize)
  periodic_bc = property(magneto.RectangularMesh.getPeriodicBC)
