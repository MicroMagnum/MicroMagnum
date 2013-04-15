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

from magnum.mesh import RectangularMesh

from .body import Body
from .shape import Everywhere
from .material import Material

class World(object):
    """
    Creates a world that is spatially discretized with the specified mesh. The constructor takes additional arguments that specifiy
    the contents of the world, i.e. subregions of the world that are associated with (possibly different) materials.

    Simple world creation with just a material:

    .. code-block:: python

      # Create a world that is entirely filled with Permalloy.
      mesh = RectangularMesh((100,100,1), (4e-9, 4e-9, 10e-9))
      world = World(mesh, Material.Py())

    World creation with different subregions ("bodies") that are each associated with an ID, a Shape and a Material:

    .. code-block:: python

      # Create a world that is composed of "Bodies"
      mesh = RectangularMesh((100,100,1), (4e-9, 4e-9, 10e-9))
      world = World(
        mesh,
        Body("the_body_id", Material.Py(), Cuboid((25e-9, 25e-9, 0e-9),
            (75e-9, 75e-9, 10e-9)))
        # (optionally more bodies, separated by commas)
      )
    """

    def __init__(self, mesh, *args):
        # Check argument types.
        if not isinstance(mesh, RectangularMesh): raise TypeError("World: The first argument to World must be a RectangularMesh instance")
        if not all(isinstance(x, (Material, Body)) for x in args): raise ValueError("World: Invalid/excess parameters in world construction.")

        self.__mesh = mesh
        self.__bodies = [x for x in args if isinstance(x, Body)]

        # Scan args for Material object.
        mat = [x for x in args if isinstance(x, Material)]
        if len(mat) == 0:
            pass
        elif len(mat) == 1:
            if len(self.__bodies) > 0: raise ValueError("World: Can't accept direct material argument if there are any Bodies specified")
            self.__bodies = [ (Body("all", mat[0], Everywhere())) ]
        else: # len(mat) > 1
            raise ValueError("World: I can accept at maximum one direct material argument")

    def __repr__(self):
        return "World(%r, %s)" % (self.mesh, ", ".join(map(repr, self.bodies)))
        #return "World@%s" % hex(id(self))

    def findBody(self, body_id):
        """
        Return the body with the given id string. It is an error if the body does not exist.

        .. code-block:: python

           body = world.findBody("the_body_id")

        """
        try:
            return [b for b in self.__bodies if b.id == body_id][0]
        except IndexError:
            raise IndexError("No body with id %s found." % body_id)

    def hasBody(self, body_id):
        try:
            self.findBody(body_id)
            return True
        except RuntimeError:
            return False

    @property
    def mesh(self):
        """
        Read-only property that returns the mesh that is associated with this world.

        .. code-block:: python

           print(world.mesh) # Retrieve the world's mesh.

        """
        return self.__mesh

    @property
    def bodies(self):
        """
        Read-only property that returns all bodies in the world.

        .. code-block:: python

           print(world.bodies)  # Retrieve a list of all world's bodies.

        """
        return self.__bodies
