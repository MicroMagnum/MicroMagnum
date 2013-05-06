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

from .storage_stephandler import StorageStepHandler

import math
try:
    import Gnuplot #, Gnuplot.PlotItems, Gnuplot.funcutils
except:
    class ImageStorage(StorageStepHandler):
        def __init__(self, output_dir, field_id_or_ids = [], filetype = "png"):
            raise NotImplementedError("ImageStorage: Functionality not supported: python 'Gnuplot' package could not be loaded.")
else:
    class ImageStorage(StorageStepHandler):
        def __init__(self, output_dir, field_id_or_ids = [], filetype = "png"):
            super(ImageStorage, self).__init__(output_dir)

            if filetype not in ["png", "jpg", "gif", "eps"]:
                raise ValueError("ImageStorage: 'filetype' must be either 'png', 'jpg', 'gif' or 'eps'")
            self.__filetype = filetype

            if hasattr(field_id_or_ids, "__iter__"):
                field_ids = list(field_id_or_ids)
            else:
                field_ids = [field_id_or_ids]
            if not all(isinstance(x, str) for x in field_ids):
                raise ValueError("ImageStorage: 'field_id' parameter must be a either a string or a collection of strings.")

            def make_filename_fn(field_id):
                return lambda state: "%s-%07i.%s" % (field_id, state.step, filetype)

            for field_id in field_ids:
                self.addVariable(field_id, make_filename_fn(field_id))

        def store(self, id, path, field, comments):
            # (comments are ignored)
            writeImage(2, "Blue-Black-Red", "x", "x", "png", field, path)

    # quantity in ["xy Angle", "xz Angle", "yz Angle", "x", "y", "z", "abs(x)", "abs(y)", "abs(z)"]:
    # palette in ["Black-White", "White-Black", "Black-White-Black", "White-Black-White", "Blue-Black-Red", "Red-Black-Blue", "Black-Blue", "Blue-Black", "Black-Red", "Red-Black", "Red-Green-Blue-Red"]:
    # filetype in ["eps", "png", "gif", "jpg"]:
    def writeImage(sub, palette, quantity, quantitystring, filetype, vectorfield, figfile):
        (nx,ny,nz) = vectorfield.mesh.num_nodes
        (dx,dy,dz) = vectorfield.mesh.delta
        dx *= 1e9
        dy *= 1e9
        dz *= 1e9

        maxM = 0
        for y in range(0, ny):
            for x in range(0, nx):
                (Mx, My, Mz) = vectorfield.get(x, y, 0)
                norm = math.sqrt(Mx*Mx+My*My+Mz*Mz)
                maxM = max(maxM, norm)

        Marray = []
        for y in range(0, ny/sub):
            for x in range(0, nx/sub):
                sumMx = 0.0
                sumMy = 0.0
                sumMz = 0.0
                num = 0.0
                for sy in range(0, sub):
                    for sx in range(0,sub):
                        (Mx, My, Mz) = vectorfield.get(sub*x+sx, sub*y+sy, 0)
                        sumMx += Mx
                        sumMy += My
                        sumMz += Mz
                        num += 1.0
                tmp = []
                tmp.append((x+0.5)*dx*sub)
                tmp.append((y+0.5)*dy*sub)
                tmp.append(sumMx/num/maxM)
                tmp.append(sumMy/num/maxM)
                tmp.append(sumMz/num/maxM)
                Marray.append(tmp)

        g = Gnuplot.Gnuplot()
        if filetype == "eps": g('set terminal postscript enhanced color "Times-Roman" 28')
        if filetype == "png": g('set terminal png enhanced size 1024, 768 giant')
        if filetype == "gif": g('set terminal gif enhanced size 1024, 768 giant')
        if filetype == "jpg": g('set terminal jpeg enhanced size 1024, 768 giant')
        g('set output "' + figfile + '"')
        g('set xrange [0:' + repr(nx*dx) + ']')
        g('set yrange [0:' + repr(ny*dy) + ']')
        g('set xlabel "x (nm)"')
        g('set ylabel "y (nm)"')
        g('set cblabel "' + quantitystring + '"')
        g('set size square')

        if quantity == "xy Angle":
            gquantity = "arg($3+{0,1}*$4)"
            g('set cbrange [-pi:pi]')
        if quantity == "xz Angle":
            gquantity = "arg($3+{0,1}*$5)"
            g('set cbrange [-pi:pi]')
        if quantity == "yz Angle":
            gquantity = "arg($4+{0,1}*$5)"
            g('set cbrange [-pi:pi]')
        if quantity == "x":
            gquantity = "$3"
            g('set cbrange [-1:1]')
        if quantity == "y":
            gquantity = "$4"
            g('set cbrange [-1:1]')
        if quantity == "z":
            gquantity = "$5"
            g('set cbrange [-1:1]')
        if quantity == "abs(x)":
            gquantity = "abs($3)"
            g('set cbrange [0:1]')
        if quantity == "abs(y)":
            gquantity = "abs($4)"
            g('set cbrange [0:1]')
        if quantity == "abs(z)":
            gquantity = "abs($5)"
            g('set cbrange [0:1]')

        if palette == "Black-White":
            g('set palette model RGB define (0 0 0 0, 1 1 1 1)')
        if palette == "White-Black":
            g('set palette model RGB define (0 1 1 1, 1 0 0 0)')
        if palette == "Black-White-Black":
            g('set palette model RGB define (0 0 0 0, 1 1 1 1, 2 0 0 0)')
        if palette == "White-Black-White":
            g('set palette model RGB define (0 1 1 1, 1 0 0 0, 2 1 1 1)')
        if palette == "Blue-Black-Red":
            g('set palette model RGB define (0 0 0 1, 0.7 0 0 0.5, 1 0 0 0, 1.3 0.5 0 0, 2 1 0 0)')
        if palette == "Red-Black-Blue":
            g('set palette model RGB define (0 1 0 0, 0.7 0.5 0 0, 1 0 0 0, 1.3 0 0 0.5, 2 0 0 1)')
        if palette == "Black-Blue":
            g('set palette model RGB define (0 0 0 0, 1 0 0 1)')
        if palette == "Blue-Black":
            g('set palette model RGB define (0 0 0 1, 1 0 0 0)')
        if palette == "Black-Red":
            g('set palette model RGB define (0 0 0 0, 1 1 0 0)')
        if palette == "Red-Black":
            g('set palette model RGB define (0 1 0 0, 1 0 0 0)')
        if palette == "Red-Green-Blue-Red":
            g('set palette model HSV define (0 0 1 1, 1 1 1 1)')

        g.plot(Gnuplot.Data(Marray, inline=1, using='($1-$3*0.7*' + repr(dx*sub) + '/2.0):($2-$4*0.7*' + repr(dy*sub) + '/2.0):($3*0.7*' + repr(dx*sub) + '):($4*0.7*' + repr(dy*sub) + '):(' + gquantity + ')', with_='vectors palette lw 1.5'))
