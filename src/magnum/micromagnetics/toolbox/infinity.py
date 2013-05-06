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
from magnum.mesh import RectangularMesh, VectorField

def calculate_strayfield(mesh, M, object_list):
    # mesh number of nodes and node size
    nx, ny, nz = mesh.num_nodes
    dx, dy, dz = mesh.delta

    # Calculate stray field for one object
    def calculate(obj, cub_M):
        cub_size   = (10e-9, 10e-9, 10e-9)
        cub_center = (0,0,0)
        cub_inf    = magneto.INFINITY_NONE
        return CalculateStrayfieldForCuboid(nx, ny, nz, dx, dy, dz, cub_M, cub_center, cub_size, cub_inf)

    # Return the sum of the stray fields of all objects.
    H = VectorField(mesh)
    H.clear()
    for obj in object_list:
        H.add(calculate(obj, M))
    return H
