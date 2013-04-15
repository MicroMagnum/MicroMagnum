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

from __future__ import print_function

from magnum import RectangularMesh, Field, VectorField

import unittest

class MyTestCase(unittest.TestCase):
    def assertVectorFieldEqual(self, H_ref, H, epsilon = 1e-4):
        for i in range(H.size()):
            v1, v2 = H_ref.get(i), H.get(i)
            dx, dy, dz = abs(v1[0]-v2[0]), abs(v1[1]-v2[1]), abs(v1[2]-v2[2])
            if dx >= epsilon or dy >= epsilon or dz >= epsilon: print("\n", i, dx, dy, dz, "\n")
            self.assertTrue(dx < epsilon and dy < epsilon and dz < epsilon)

def right_rotate_vector_field(M):
    pbc, pbc_rep = M.mesh.periodic_bc
    pbc2, pbc_rep2 = "", pbc_rep
    if "x" in pbc: pbc2 += "y"
    if "y" in pbc: pbc2 += "z"
    if "z" in pbc: pbc2 += "x"

    nn = M.mesh.num_nodes
    dd = M.mesh.delta
    mesh = RectangularMesh((nn[2], nn[0], nn[1]), (dd[2], dd[0], dd[1]), pbc2, pbc_rep2)

    M2 = VectorField(mesh)
    for x, y, z in M.mesh.iterateCellIndices():
        a = M.get(x,y,z)
        M2.set(z,x,y, (a[2], a[0], a[1]))
    return M2

def left_rotate_vector_field(M):
    pbc, pbc_rep = M.mesh.periodic_bc
    pbc2, pbc_rep2 = "", pbc_rep
    if "x" in pbc: pbc2 += "z"
    if "y" in pbc: pbc2 += "x"
    if "z" in pbc: pbc2 += "y"

    nn = M.mesh.num_nodes
    dd = M.mesh.delta
    mesh = RectangularMesh((nn[1], nn[2], nn[0]), (dd[1], dd[2], dd[0]), pbc2, pbc_rep2)

    M2 = VectorField(mesh)
    for x, y, z in M.mesh.iterateCellIndices():
        a = M.get(x,y,z)
        M2.set(y,z,x, (a[1], a[2], a[0]))
    return M2

def right_rotate_field(M):
    pbc, pbc_rep = M.mesh.periodic_bc
    pbc2, pbc_rep2 = "", pbc_rep
    if "x" in pbc: pbc2 += "y"
    if "y" in pbc: pbc2 += "z"
    if "z" in pbc: pbc2 += "x"

    nn = M.mesh.num_nodes
    dd = M.mesh.delta
    mesh = RectangularMesh((nn[2], nn[0], nn[1]), (dd[2], dd[0], dd[1]), pbc2, pbc_rep2)

    M2 = Field(mesh)
    for x, y, z in M.mesh.iterateCellIndices():
        a = M.get(x,y,z)
        M2.set(z,x,y, a)
    return M2
