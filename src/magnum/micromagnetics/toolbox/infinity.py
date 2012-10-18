import magnum.magneto as magneto
from magnum.meshes import RectangularMesh, VectorField

def calculate_strayfield(mesh, M, object_list):
  # mesh number of nodes and node size
  nx, ny, nz = mesh.num_nodes
  dx, dy, dz = mesh.delta

  # Calculate stray field for one object
  def calculate(obj, cub_M):
    cub_size   = (10e-9, 10e-9, 10e-9)
    cub_center = (0,0,0)
    cub_inf    = magneto.INFINITY_NONE
    return CalculateStrayfieldForCuboid(nx, ny, nz, dx, dy, dz, cub_M, cub_center, cub_size, cub_inf)

  # Return the sum of the stray fields of all objects.
  H = VectorField(mesh)
  H.clear()
  for obj in object_list:
    H.add(calculate(obj, M))
  return H
