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

from magnum.micromagnetics.io.write_omf import writeOMF, OMF_FORMAT_ASCII, OMF_FORMAT_BINARY_4, OMF_FORMAT_BINARY_8
from magnum.micromagnetics.io.read_omf import readOMF
from magnum.micromagnetics.io.write_vtk import writeVTK
from magnum.micromagnetics.io.write_image import writeImage, createImage

__all__ = [
    "writeOMF", "OMF_FORMAT_ASCII", "OMF_FORMAT_BINARY_4", "OMF_FORMAT_BINARY_8",
    "readOMF",
    "writeVTK",
    "writeImage", "createImage",
]
