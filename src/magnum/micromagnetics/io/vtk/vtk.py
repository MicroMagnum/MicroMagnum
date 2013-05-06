#
# Adapted from the PyEVtk library, https://bitbucket.org/pauloh/pyevtk
#
# Copyright of the original code is replicated below.
#

# ***********************************************************************************
# * Copyright 2010 Paulo A. Herrera. All rights reserved.                           *
# *                                                                                 *
# * Redistribution and use in source and binary forms, with or without              *
# * modification, are permitted provided that the following conditions are met:     *
# *                                                                                 *
# *  1. Redistributions of source code must retain the above copyright notice,      *
# *  this list of conditions and the following disclaimer.                          *
# *                                                                                 *
# *  2. Redistributions in binary form must reproduce the above copyright notice,   *
# *  this list of conditions and the following disclaimer in the documentation      *
# *  and/or other materials provided with the distribution.                         *
# *                                                                                 *
# * THIS SOFTWARE IS PROVIDED BY PAULO A. HERRERA ``AS IS'' AND ANY EXPRESS OR      *
# * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF    *
# * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO      *
# * EVENT SHALL <COPYRIGHT HOLDER> OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,        *
# * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,  *
# * BUT NOT LIMITED TO, PROCUREMEN OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,    *
# * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY           *
# * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING  *
# * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS              *
# * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                    *
# ***********************************************************************************

# **************************************
# *  Low level Python library to       *
# *  export data to binary VTK file.   *
# **************************************

from .xml import XmlWriter

import sys
import os
import struct

# ================================
#            VTK Types
# ================================

#     FILE TYPES
class VtkFileType(object):

    def __init__(self, name, ext):
        self.name = name
        self.ext  = ext

    def __str__(self):
        return "Name: %s  Ext: %s \n" % (self.name, self.ext)

VtkImageData        = VtkFileType("ImageData", ".vti")
VtkPolyData         = VtkFileType("PolyData", ".vtp")
VtkRectilinearGrid  = VtkFileType("RectilinearGrid", ".vtr")
VtkStructuredGrid   = VtkFileType("StructuredGrid", ".vts")
VtkUnstructuredGrid = VtkFileType("UnstructuredGrid", ".vtu")

#    DATA TYPES
class VtkDataType(object):

    def __init__(self, size, name):
        self.size = size
        self.name = name

    def __str__(self):
        return "Type: %s  Size: %d \n" % (self.name, self.size)

VtkInt8    = VtkDataType(1, "Int8")
VtkUInt8   = VtkDataType(1, "UInt8")
VtkInt16   = VtkDataType(2, "Int16")
VtkUInt16  = VtkDataType(2, "UInt16")
VtkInt32   = VtkDataType(4, "Int32")
VtkUInt32  = VtkDataType(4, "UInt32")
VtkInt64   = VtkDataType(8, "Int64")
VtkUInt64  = VtkDataType(8, "UInt64")
VtkFloat32 = VtkDataType(4, "Float32")
VtkFloat64 = VtkDataType(8, "Float64")

#    CELL TYPES
class VtkCellType(object):

    def __init__(self, tid, name):
        self.tid = tid
        self.name = name

    def __str__(self):
        return "VtkCellType( %s ) \n" % ( self.name )

VtkVertex = VtkCellType(1, "Vertex")
VtkPolyVertex = VtkCellType(2, "PolyVertex")
VtkLine = VtkCellType(3, "Line")
VtkPolyLine = VtkCellType(4, "PolyLine")
VtkTriangle = VtkCellType(5, "Triangle")
VtkTriangleStrip = VtkCellType(6, "TriangleStrip")
VtkPolygon = VtkCellType(7, "Polygon")
VtkPixel = VtkCellType(8, "Pixel")
VtkQuad = VtkCellType(9, "Quad")
VtkTetra = VtkCellType(10, "Tetra")
VtkVoxel = VtkCellType(11, "Voxel")
VtkHexahedron = VtkCellType(12, "Hexahedron")
VtkWedge = VtkCellType(13, "Wedge")
VtkPyramid = VtkCellType(14, "Pyramid")
VtkQuadraticEdge = VtkCellType(21, "Quadratic_Edge")
VtkQuadraticTriangle = VtkCellType(22, "Quadratic_Triangle")
VtkQuadraticQuad = VtkCellType(23, "Quadratic_Quad")
VtkQuadraticTetra = VtkCellType(24, "Quadratic_Tetra")
VtkQuadraticHexahedron = VtkCellType(25, "Quadratic_Hexahedron")

# ==============================
#       Helper functions
# ==============================
def _mix_extents(start, end):
    assert (len(start) == len(end) == 3)
    string = "%d %d %d %d %d %d" % (start[0], end[0], start[1], end[1], start[2], end[2])
    return string

def _array_to_string(a):
    s = " ".join(list(a))
    return s

def _get_byte_order():
    if sys.byteorder == "little":
        return "LittleEndian"
    else:
        return "BigEndian"

# ================================
#        VtkGroup class
# ================================
class VtkGroup(object):

    def __init__(self, rootpath = ".", filename = "group.pvd"):
        """ Creates a VtkGroup file that is stored in filepath.

            PARAMETERS:
                filepath: filename (should include .pvd extension)
        """
        self.xml = XmlWriter(os.path.normpath(rootpath + "/" + filename))
        self.xml.openElement("VTKFile")
        self.xml.addAttributes(type = "Collection", version = "0.1",  byte_order = _get_byte_order())
        self.xml.openElement("Collection")
        self.root = os.path.normpath(rootpath)

    def save(self):
        """ Closes this VtkGroup. """
        self.xml.closeElement("Collection")
        self.xml.closeElement("VTKFile")
        self.xml.close()

    def addFile(self, filepath, **attributes):
        """ Adds file to this VTK group.

            PARAMETERS:
                filepath: full path to VTK file.
                sim_time: simulated time.
        """
        # TODO: Check what the other attributes are for.

        filename = os.path.normpath(filepath)
        self.xml.openElement("DataSet")
        self.xml.addAttributes(group = "", part = "0", file = filename, **attributes)
        self.xml.closeElement()



# ================================
#        VtkFile class
# ================================
class VtkFile(object):

    def __init__(self, filepath, ftype):
        """
            PARAMETERS:
                filepath: filename without extension.
                ftype: file type, e.g. VtkImageData, etc.
        """
        self.ftype = ftype
        self.filename = filepath
        self.xml = XmlWriter(self.filename)
        self.offset = 0  # offset in bytes after beginning of binary section
        self.appendedDataIsOpen = False

        self.xml.openElement("VTKFile").addAttributes(type = ftype.name,
                                                      version = "0.1",
                                                      byte_order = _get_byte_order())

    def getFileName(self):
        """ Returns absolute path to this file. """
        return  os.path.abspath(self.filename)

    def openPiece(self, start = None, end = None,
                        npoints = None, ncells = None,
                        nverts = None, nlines = None, nstrips = None, npolys = None):
        """ Open piece section.

            PARAMETERS:
                Next two parameters must be given together.
                start: array or list with start indexes in each direction.
                end:   array or list with end indexes in each direction.

                npoints: number of points in piece (int).
                ncells: number of cells in piece (int). If present,
                        npoints must also be given.

                All the following parameters must be given together with npoints.
                They should all be integer values.
                nverts: number of vertices.
                nlines: number of lines.
                nstrips: number of strips.
                npolys: number of .

            RETURNS:
                this VtkFile to allow chained calls.
        """
        # TODO: Check what are the requirements for each type of grid.

        self.xml.openElement("Piece")
        if (start and end):
            ext = _mix_extents(start, end)
            self.xml.addAttributes( Extent = ext)

        elif (ncells and npoints):
            self.xml.addAttributes(NumberOfPoints = npoints, NumberOfCells = ncells)

        elif (npoints and nverts and nlines and nstrips and npolys):
            self.xml.addAttributes(npoints = npoints, nverts = nverts,
                    nlines = nlines, nstrips = nstrips, npolys = npolys)
        else:
            assert(False)

        return self

    def closePiece(self):
        self.xml.closeElement("Piece")

    def openData(self, nodeType, scalars=None, vectors=None, normals=None, tensors=None, tcoords=None):
        """ Open data section.

            PARAMETERS:
                nodeType: Point or Cell.
                scalars: default data array name for scalar data.
                vectors: default data array name for vector data.
                normals: default data array name for normals data.
                tensors: default data array name for tensors data.
                tcoords: dafault data array name for tcoords data.

            RETURNS:
                this VtkFile to allow chained calls.
        """
        self.xml.openElement(nodeType + "Data")
        if scalars:
            self.xml.addAttributes(scalars = scalars)
        if vectors:
            self.xml.addAttributes(vectors = vectors)
        if normals:
            self.xml.addAttributes(normals = normals)
        if tensors:
            self.xml.addAttributes(tensors = tensors)
        if tcoords:
            self.xml.addAttributes(tcoords = tcoords)

        return self

    def closeData(self, nodeType):
        """ Close data section.

            PARAMETERS:
                nodeType: Point or Cell.

            RETURNS:
                this VtkFile to allow chained calls.
        """
        self.xml.closeElement(nodeType + "Data")


    def openGrid(self, start = None, end = None, origin = None, spacing = None):
        """ Open grid section.

            PARAMETERS:
                start: array or list of start indexes. Required for Structured, Rectilinear and ImageData grids.
                end: array or list of end indexes. Required for Structured, Rectilinear and ImageData grids.
                origin: 3D array or list with grid origin. Only required for ImageData grids.
                spacing: 3D array or list with grid spacing. Only required for ImageData grids.

            RETURNS:
                this VtkFile to allow chained calls.
        """
        gType = self.ftype.name
        self.xml.openElement(gType)
        if (gType == VtkImageData.name):
            if (not start or not end or not origin or not spacing): assert(False)
            ext = _mix_extents(start, end)
            self.xml.addAttributes(WholeExtent = ext,
                                   Origin = _array_to_string(origin),
                                   Spacing = _array_to_string(spacing))

        elif (gType == VtkStructuredGrid.name or gType == VtkRectilinearGrid.name):
            if (not start or not end): assert (False)
            ext = _mix_extents(start, end)
            self.xml.addAttributes(WholeExtent = ext)

        return self

    def closeGrid(self):
        """ Closes grid element.

            RETURNS:
                this VtkFile to allow chained calls.
        """
        self.xml.closeElement(self.ftype.name)


    def addData(self, name, dtype, nelem, ncomp):
        """ Adds data array description to xml header section.

            PARAMETERS:
                name: data array name.
                dtype: type of the data, e.g. VtkFloat64
                nelem: number of elements in the array.
                ncomp: number of components, 1 (=scalar) and 3 (=vector).

            RETURNS:
                This VtkFile to allow chained calls.
        """

        self.xml.openElement( "DataArray")
        self.xml.addAttributes( Name = name,
                                NumberOfComponents = ncomp,
                                type = dtype.name,
                                format = "appended",
                                offset = self.offset)
        self.xml.closeElement()

        #TODO: Check if 4 is platform independent
        self.offset += nelem * ncomp * dtype.size + 4 # add 4 to indicate array size
        return self

    def appendData(self, data):
        """ Append data to binary section.
            This function writes the header section and the data to the binary file.

            PARAMETERS:
                data: the bytearray that holds the data to write. must have the correct length

            RETURNS:
                This VtkFile to allow chained calls
        """
        self.openAppendedData()

        self.xml.stream.write(struct.pack('=i', len(data)))
        self.xml.stream.write(data)

        return self

    def openAppendedData(self):
        """ Opens binary section.

            It is not necessary to explicitly call this function from an external library.
        """
        if not self.appendedDataIsOpen:
            self.xml.openElement("AppendedData").addAttributes(encoding = "raw").addText("_")
            self.appendedDataIsOpen = True

    def closeAppendedData(self):
        """ Closes binary section.

            It is not necessary to explicitly call this function from an external library.
        """
        self.xml.closeElement("AppendedData")
        self.appendedDataIsOpen = False

    def openElement(self, tagName):
        """ Useful to add elements such as: Coordinates, Points, Verts, etc. """
        self.xml.openElement(tagName)

    def closeElement(self, tagName):
        self.xml.closeElement(tagName)

    def save(self):
        """ Closes file """
        if self.appendedDataIsOpen:
            self.xml.closeElement("AppendedData")
        self.xml.closeElement("VTKFile")
        self.xml.close()
