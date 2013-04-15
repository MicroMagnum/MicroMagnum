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

class RectangularMesh(magneto.RectangularMesh):

    def __init__(self, num_nodes, cell_size, periodic_bc="", periodic_repeat=None):

        nx, ny, nz = map(  int, num_nodes)
        dx, dy, dz = map(float, cell_size)

        if not periodic_repeat:
            # count number of periodic directions (0, 1, 2, or 3)
            num_dirs = sum(1 for s in ("x", "y", "z") if periodic_bc.find(s) != -1)
            # defaults for peri_repeat for counted num_dirs.
            periodic_repeat = {0:1, 1:15, 2:3, 3:2}[num_dirs]

        super(RectangularMesh, self).__init__(nx, ny, nz, dx, dy, dz, periodic_bc, periodic_repeat)
