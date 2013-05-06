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

from magnum.config import cfg
from magnum.logger import logger

class TensorField(object):
    PADDING_DISABLE             = magneto.PADDING_DISABLE
    PADDING_ROUND_2             = magneto.PADDING_ROUND_2
    PADDING_ROUND_4             = magneto.PADDING_ROUND_4
    PADDING_ROUND_8             = magneto.PADDING_ROUND_8
    PADDING_ROUND_POT           = magneto.PADDING_ROUND_POT
    PADDING_SMALL_PRIME_FACTORS = magneto.PADDING_SMALL_PRIME_FACTORS

    def __init__(self, mesh, padding):
        self.__mesh = mesh
        self.__padding = padding
        self.setPeriodicBoundaries(False, False, False, 0)

    def setPeriodicBoundaries(self, x, y, z, repeat=3):
        self.__pbc_x, self.__pbc_y, self.__pbc_z = x, y, z
        self.__pbc_repeat = repeat

    def getPeriodicBoundaries(self):
        return self.__pbc_x, self.__pbc_y, self.__pbc_z, self.__pbc_repeat

    padding = property(lambda self: self.__padding)
    mesh    = property(lambda self: self.__mesh)

    def generate(self):
        raise NotImplementedError("TensorField.generate")

class DemagTensorField(TensorField):
    def __init__(self, mesh, padding = TensorField.PADDING_ROUND_4):
        super(DemagTensorField, self).__init__(mesh, padding)

    def generate(self):
        nx, ny, nz = self.mesh.num_nodes
        dx, dy, dz = self.mesh.delta
        pbc_x, pbc_y, pbc_z, pbc_repeat = self.getPeriodicBoundaries()

        N = magneto.GenerateDemagTensor(
            nx, ny, nz,
            dx, dy, dz,
            pbc_x, pbc_y, pbc_z, pbc_repeat,
            self.padding,
            cfg.global_cache_directory
        )
        return N

class PhiTensorField(TensorField):
    def __init__(self, mesh, padding = TensorField.PADDING_ROUND_4):
        super(PhiTensorField, self).__init__(mesh, padding)

    def generate(self):
        nx, ny, nz = self.mesh.num_nodes
        dx, dy, dz = self.mesh.delta
        pbc_x, pbc_y, pbc_z, pbc_repeat = self.getPeriodicBoundaries()

        N = magneto.GeneratePhiDemagTensor(
            nx, ny, nz,
            dx, dy, dz,
            pbc_x, pbc_y, pbc_z, pbc_repeat,
            self.padding,
            magnum_config.global_cache_directory
        )
        return N

class StrayFieldCalculator(object):
    def __init__(self, mesh, method = "tensor", padding = TensorField.PADDING_ROUND_4):
        # are we periodic?
        peri, peri_repeat = mesh.periodic_bc
        peri_x = peri.find("x") != -1
        peri_y = peri.find("y") != -1
        peri_z = peri.find("z") != -1

        # number of cells and cell sizes
        nx, ny, nz = mesh.num_nodes
        dx, dy, dz = mesh.delta

        if nx * ny * nz > 32:
            if not (nx < ny < nz):
                logger.info("Performance hint: The number of cells nx, ny, nz in each direction should satisfy nx >= ny >= nz.")
            if (nx == 1 or ny == 1) and nz != 1:
                logger.info("Performance hint: Meshes with 2-dimensional cell grids should span the xy-plane, i.e. the number of cells in z-direction should be 1.")

        # generate calculation function depending on user-selected method
        if method == "tensor":
            tensor = DemagTensorField(mesh, padding)
            tensor.setPeriodicBoundaries(peri_x, peri_y, peri_z, peri_repeat)

            # Determine if we should use the fast convolution (via FFT) or the simple convolution (using for-loops, CPU only).
            if cfg.isCudaEnabled():
                use_fft = True
            else:
                # HACK: Our CPU implementation of fast convolution doesn't support these input dimensions.
                if nx == 1 and (ny != 1 or nz != 1):
                    use_fft = False
                else:
                    use_fft = (nx * ny * nz >= 32) # this is the break-even point for using FFT convolutions on my system.

            if use_fft:
                conv = magneto.SymmetricMatrixVectorConvolution_FFT(tensor.generate(), nx, ny, nz)
            else:
                conv = magneto.SymmetricMatrixVectorConvolution_Simple(tensor.generate(), nx, ny, nz)
            self.__calc = lambda M, H: conv.execute(M, H)

        elif method == "potential":
            assert not peri_x and not peri_y and not peri_z
            tensor = PhiTensorField(mesh, padding)
            tensor.setPeriodicBoundaries(peri_x, peri_y, peri_z, peri_repeat)
            conv = magneto.VectorVectorConvolution_FFT(tensor.generate(), nx, ny, nz, dx, dy, dz)
            self.__calc = lambda M, H: conv.execute(M, H)
        # experimental, don't use (and not implemented :) )
        elif method == "single":
            assert not peri_x and not peri_y and not peri_z
            stray = magneto.StrayField_single(nx, ny, nz, dx, dy, dz)
            self.__calc = lambda M, H: stray.calculate(M, H)
        elif method == "multi":
            assert not peri_x and not peri_y and not peri_z
            stray = magneto.StrayField_multi(nx, ny, nz, dx, dy, dz)
            self.__calc = lambda M, H: stray.calculate(M, H)
        else:
            assert False

    def calculate(self, M, H):
        self.__calc(M, H)
