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

from .vtk import VtkFile, VtkRectilinearGrid, VtkFloat64
import struct

def writeVTK(filename, field):
    mesh = field.mesh
    n = mesh.num_nodes
    d = mesh.delta

    # I. Describe data entries in file
    start, end = (0, 0, 0), (n[0], n[1], n[2])
    w = VtkFile(filename, VtkRectilinearGrid)
    w.openGrid(start = start, end = end)
    w.openPiece(start = start, end = end)

    # - Magnetization data
    w.openData("Cell", vectors = "M")
    w.addData("M", VtkFloat64, field.size(), 3)
    w.closeData("Cell")

    # - Coordinate data
    w.openElement("Coordinates")
    w.addData("x_coordinate", VtkFloat64, n[0] + 1, 1)
    w.addData("y_coordinate", VtkFloat64, n[1] + 1, 1)
    w.addData("z_coordinate", VtkFloat64, n[2] + 1, 1)
    w.closeElement("Coordinates")

    w.closePiece()
    w.closeGrid()

    # II. Append binary parts to file
    def coordRange(start, step, n):
        result = bytearray(0)
        for i in range(0, n+1):
            result = result + struct.pack('d', start + step * i)
        return result

    w.appendData(field.toByteArray())
    w.appendData(coordRange(0.0, d[0], n[0]))
    w.appendData(coordRange(0.0, d[1], n[1]))
    w.appendData(coordRange(0.0, d[2], n[2]))

    # III. Save & close
    w.save()
