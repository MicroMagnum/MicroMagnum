import magnum.magneto as magneto

from magnum.logger import logger
from magnum.meshes import RectangularMesh, VectorField

def readOMF(path):
  # Read OMF file
  header = magneto.OMFHeader()
  mat = magneto.readOMF(path, header)

  # Convert (header, mat) to VectorField.
  mesh = RectangularMesh((header.xnodes, header.ynodes, header.znodes), (header.xstepsize, header.ystepsize, header.zstepsize))
  vector_field = VectorField(mesh, id=None, value_unit=header.valueunit)
  vector_field.assign(mat)

  logger.debug("Read file %s", path)
  return vector_field
