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

from .shape import Shape

from magnum.logger import logger
from magnum.mesh import RectangularMesh
from math import log, floor, ceil

try:
    import gmshpy
    _found_gmsh_lib = True
except:
    _found_gmsh_lib = False
    logger.warn("Python wrappers for GMSH not found!")
    logger.warn("-> This means that the GmshShape class is not available!")

class GmshShape(Shape):
    """
    This class can create a shape from a number of gemetry/mesh files that are supported by GMSH.
    The Python wrappers for GMSH have to be installed on the system.
    """
    def __init__(self, model, shift = (0.0, 0.0, 0.0), scale = 1.0):
        if not _found_gmsh_lib: raise NotImplementedError("GmshShape class can not be used because the Python wrappers for GMSH could not be loaded ('import gmshpy')")
        super(GmshShape, self).__init__()

        self.__model = model
        self.__shift = shift
        self.__scale = scale

    def isPointInside(self, pt):
        spt = gmshpy.SPoint3(
            pt[0] / self.__scale + self.__shift[0],
            pt[1] / self.__scale + self.__shift[1],
            pt[2] / self.__scale + self.__shift[2])
        return self.__model.getMeshElementByCoord(spt)

    @staticmethod
    def with_mesh_from_file(filename, cell_size, scale = 1.0):
        if not _found_gmsh_lib: raise NotImplementedError("GmshShape class can not be used because the Python wrappers for GMSH could not be loaded ('import gmshpy')")

        # Create GMSH model
        model = gmshpy.GModel()
        model.setFactory("Gmsh")
        model.load(filename)

        # mesh if not already meshed
        if model.getMeshStatus() < 3: model.mesh(3)

        # get bounds
        bounds = model.bounds()
        p1 = (bounds.min().x(), bounds.min().y(), bounds.min().z())
        p2 = (bounds.max().x(), bounds.max().y(), bounds.max().z())

        # Estimate cell size h and refine mesh accordingly
        bb_volume   = reduce(lambda x,y: x*y, ((b-a) for a,b in zip(p1, p2)))
        h_target    = min(cell_size) / scale
        h_est       = pow(bb_volume / model.getNumMeshElements(), 1.0/3.0)
        refinements = int(ceil(log(h_est / h_target, 2)))
        for i in range(0, refinements): model.refineMesh(0)

        # Create Rectangular Mesh
        num_nodes = [int(ceil((b-a)/c*scale)) for a,b,c in zip(p1, p2, cell_size)]
        mesh = RectangularMesh(num_nodes, cell_size)

        return mesh, GmshShape(model, p1, scale)
