from .vtk import *

def writeVTK(filename, field):
  mesh = field.mesh
  n = mesh.getFieldMatrixDimensions()
  d = mesh.delta

  # I. Describe data entries in file
  start, end = (0, 0, 0), (n[0], n[1], n[2])
  w = VtkFile(filename, VtkRectilinearGrid)
  w.openGrid(start = start, end = end)
  w.openPiece(start = start, end = end)

  # - Magnetization data
  w.openData("Cell", vectors = "M")
  w.addData("M", VtkFloat64, field.size(), 3)
  w.closeData("Cell")

  # - Coordinate data
  w.openElement("Coordinates")
  w.addData("x_coordinate", VtkFloat64, n[0]+1, 1)
  w.addData("y_coordinate", VtkFloat64, n[1]+1, 1)
  w.addData("z_coordinate", VtkFloat64, n[2]+1, 1)
  w.closeElement("Coordinates")

  w.closePiece()
  w.closeGrid()

  # II. Append binary parts to file
  def coordRange(start, step, n):
    result = bytearray(0)
    for i in range(0, n+1):
      result = result + struct.pack('d', start + step * i)
    return result

  w.appendData(field.toByteArray())
  w.appendData(coordRange(0.0, d[0], n[0]))
  w.appendData(coordRange(0.0, d[1], n[1]))
  w.appendData(coordRange(0.0, d[2], n[2]))

  # III. Save & close
  w.save()
