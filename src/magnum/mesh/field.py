import magnum.magneto as magneto

from .rectangular_mesh import RectangularMesh

class Field(magneto.Matrix):
  def __init__(self, mesh, id = None, value_unit = None):
    super(Field, self).__init__(magneto.Shape(*mesh.getFieldMatrixDimensions()))
    self.__mesh = mesh
    self.__id = id
    self.__value_unit = value_unit

  def __repr__(self):
    return "Field(%r, %r, %r)" % (self.shape, self.__id, self.__value_unit)

  def getMesh(self): return self.__mesh
  mesh = property(getMesh)

  def interpolate(self, mesh):
    if not isinstance(mesh, RectangularMesh):
      raise TypeError("Field.interpolate: Fixme: Non-rectangular meshes are not supported!")

    # Get matrix with interpolated values in 'interp_mat'   
    need_interpolate = (self.mesh.num_nodes != mesh.num_nodes)
    if need_interpolate:
      nx, ny, nz = mesh.num_nodes # new size (in number of cells)
      interp_mat = magneto.linearInterpolate(self, magneto.Shape(nx, ny, nz))
    else:
      interp_mat = self # no need to interpolate..

    # Create interpolated vector field from matrix 'interp_mat'
    result = Field(mesh)
    result.assign(interp_mat)
    return result
