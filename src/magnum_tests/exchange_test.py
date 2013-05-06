#!/usr/bin/python

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

from magnum import *
import magnum.magneto as magneto

from .helpers import MyTestCase, right_rotate_vector_field, right_rotate_field, left_rotate_vector_field

import unittest
import random
import itertools

class ExchangeTest(MyTestCase):

    def do_test(self, M_file, H_file, epsilon):
        # load ref
        M     = readOMF(M_file)
        H_ref = readOMF(H_file)
        H     = VectorField(M.mesh)
        A     = Field(M.mesh); A.fill(Material.Py().A)
        Ms    = Field(M.mesh); Ms.fill(Material.Py().Ms)

        # calculate
        mesh = M.mesh
        nx, ny, nz = mesh.num_nodes
        dx, dy, dz = mesh.delta
        bcx, bcy, bcz = False, False, False
        magneto.exchange(nx, ny, nz, dx, dy, dz, bcx, bcy, bcz, Ms, A, M, H)

        # compare
        self.assertVectorFieldEqual(H_ref, H, epsilon)

    def test_calculate_2d(self):
        self.do_test("ref/M1.omf", "ref/H1_exch.omf", 2e0)

    def test_calculate_3d(self):
        self.do_test("ref/M3.omf", "ref/H3_exch.ohf", 1e2)

    def pbc_exchange(self, nx, ny, nz):
        dx, dy, dz = 1e-9, 1e-9, 1e-9

        mesh = RectangularMesh((nx, ny, nz), (dx, dy, dz))
        A  = Field(mesh); A.fill(Material.Py().A)
        Ms = Field(mesh); Ms.fill(Material.Py().Ms)
        M  = VectorField(mesh)
        H  = VectorField(mesh)

        for bcx, bcy, bcz in itertools.product([False, True], [False, True], [False, True]):
            M.fill(tuple((random.random()-0.5) * 2e5 for i in range(3)))
            #M.fill((8e5, 8e5, -8e5))
            magneto.exchange(nx, ny, nz, dx, dy, dz, bcx, bcy, bcz, Ms, A, M, H)
            for i in range(nx*ny*nz):
                self.assertEquals(H.get(i), (0.0, 0.0, 0.0))

    def test_pbc_exchange_3d(self):
        self.pbc_exchange(64, 64, 64) # mod 8 == 0 (cuda thread block size)
        self.pbc_exchange(64+6, 64-4, 64-5)

    def test_pbc_exchange_2d(self):
        self.pbc_exchange(16, 16, 1) # mod 16 == 0 (cuda thread block size)
        self.pbc_exchange(16-5, 16+3, 1)

    def test_rotated_magnetization_produces_same_rotated_exchange_field(self):

        def compute(M, Ms, A, rotations=1):
            for _ in range(rotations):
                M = right_rotate_vector_field(M)
                Ms = right_rotate_field(Ms); A = right_rotate_field(A)
            H = VectorField(M.mesh)
            nx, ny, nz = M.mesh.num_nodes
            dx, dy, dz = M.mesh.delta
            pbc, pbc_rep = M.mesh.periodic_bc
            bcx, bcy, bcz = "x" in pbc, "y" in pbc, "z" in pbc
            magneto.exchange(nx, ny, nz, dx, dy, dz, bcx, bcy, bcz, Ms, A, M, H)
            for _ in range(rotations): H = left_rotate_vector_field(H)
            return H

        #mesh = RectangularMesh((32,16,8), (1e-9,1e-9,1e-9), "z", 20)
        mesh = RectangularMesh((20, 20, 6), (5e-9, 5e-9, 5e-9), "xy", 20)
        M0 = VectorField(mesh); M0.randomize(); M0.scale(8e5)
        A  = Field(mesh);  A.fill(Material.Py().A)
        Ms = Field(mesh); Ms.fill(Material.Py().Ms)

        H0 = compute(M0, Ms, A, 0)
        H1 = compute(M0, Ms, A, 1)
        H2 = compute(M0, Ms, A, 2)

        self.assertVectorFieldEqual(H0, H1, 1e0)
        self.assertVectorFieldEqual(H0, H2, 1e0)

if __name__ == '__main__':
    unittest.main()
