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

from magnum.logger import logger
from magnum.mesh import RectangularMesh, VectorField

def readOMF(path):
    # Read OMF file
    header = magneto.OMFHeader()
    mat = magneto.readOMF(path, header)

    # Convert (header, mat) to VectorField.
    mesh = RectangularMesh((header.xnodes, header.ynodes, header.znodes), (header.xstepsize, header.ystepsize, header.zstepsize))
    vector_field = VectorField(mesh)
    vector_field.assign(mat)

    logger.debug("Read file %s", path)
    return vector_field
