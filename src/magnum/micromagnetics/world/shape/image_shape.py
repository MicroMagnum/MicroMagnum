# Copyright 2012 by the Micromagnum authors.
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

try:
  import Image
  _found_image_lib = True
except:
  _found_image_lib = False
  logger.warn("Python Imaging Library not found!")
  logger.warn("-> This means that the ImageShapeCreator and related classes are not available!")

class ImageShape(Shape):
  """
  This class can create a shape with an image. The image defines the shape between the x-y-axes and z is filled in the whole world.

  To use the ImageShape take the *ImageShapeCreator*. For example:

  .. code-block:: python
    
      isc = ImageShapeCreator("phi.png", mesh)
      world = World(mesh, Body("phi", Material.Py(), isc.pick("black")))

  You can use the following colors:
  *black, green, blue, red, yellow, grey, white.*

  It is not necessary to use a color exactly. Every pixel is interpreted as the nearest color.

  """
  def __init__(self, isc, color):
    if not _found_image_lib: raise NotImplementedError("ImageShape class is not implemented because the Python Imaging Library was not found")
    super(ImageShape, self).__init__()
    self.__isc = isc
    self.__test_fn = isc.makeColorTestFunction(color)

  def getCellIndices(self, mesh):
    return self.__isc.getCellIndices(mesh, self.__test_fn)

  def isPointInside(self, pt):
    return self.__isc.isPointInside(self.__test_fn, pt)
  

class ImageShapeCreator(object):
  def __init__(self, filename, mesh):
    if not _found_image_lib: raise NotImplementedError("ImageShapeCreator class is not implemented because the Python Imaging Library was not found")
    if not isinstance(mesh, RectangularMesh): raise ValueError("ImageShapeCreator: 'mesh' argument must be a RectangularMesh object")
    if not isinstance(filename, str): raise ValueError("ImageShapeCreator: 'filename' argument must be a string")

    # get image object from filename
    image = Image.open(filename).convert("RGB")

    self.__image = image
    self.__mesh = mesh
    self.__imgsize = image.size
    self.__pixelsize = (mesh.size[0] / float(image.size[0]), mesh.size[1] / float(image.size[1]), mesh.size[2]) # "size" of a pixel in meter

  def pick(self, color = "black"):
    return ImageShape(self, color)

  def makeColorTestFunction(self, p): # p: pick-color
    coltab = {
      "black": (0,0,0),
      "green": (0,255,0),
      "blue":  (0,0,255),
      "red":   (255,0,0),
      "yellow":(255,255,0),
      "grey":  (127,127,127),
      "white": (255,255,255)
    }

    if type(p) == str: 
      try:
        p = coltab[p]
      except KeyError:
        raise ValueError("Need a known color identifier")
    elif type(p) == tuple:
      if len(p) != 3:
        raise ValueError("Need a RGB tuple with three entries to specify the pick color: (r,g,b) in {0..255} x {0..255} x {0..255}")
    else:
      raise ValueError("Need either a string color identifier or a RGB tuple (r,g,b) in {0..255} x {0..255} x {0..255}.")

    def fn(c):
      d = (c[0]-p[0], c[1]-p[1], c[2]-p[2])
      return d[0]*d[0]+d[1]*d[1]+d[2]*d[2] < 16*16
    return fn

  def isPointInside(self, test_fn, pos):
    image     = self.__image
    pixelsize = self.__pixelsize
    imgsize   = self.__imgsize

    # Get corresponding column and row in image
    col =              round(pos[0] / pixelsize[0] - 0.5)
    row = imgsize[1] - round(pos[1] / pixelsize[1] + 0.5) # (0,0) is bottom-left of image.

    # Check if col, row is in image
    if col < 0 or col >= imgsize[0]: return False
    if row < 0 or row >= imgsize[1]: return False

    # Check if position is marked in image
    return test_fn(image.getpixel((col, row)))

  def getCellIndices(self, mesh, test_fn):
    image     = self.__image
    pixelsize = self.__pixelsize
    imgsize   = self.__imgsize

    result = []

    for i in range(0, mesh.total_nodes):
      pos = mesh.getPosition(i)

      # Get corresponding column and row in image
      col =              round(pos[0] / pixelsize[0] - 0.5)
      row = imgsize[1] - round(pos[1] / pixelsize[1] + 0.5)

      # Check if col, row is in image
      if col < 0 or col >= imgsize[0]: continue
      if row < 0 or row >= imgsize[1]: continue

      # Append cell if marked in image
      if test_fn(image.getpixel((col, row))):
        result.append(i)

    return result
