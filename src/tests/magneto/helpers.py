from magnum import *

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
  for x,y,z in M.mesh.iterateCellIndices(): 
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
  for x,y,z in M.mesh.iterateCellIndices():
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
  for x,y,z in M.mesh.iterateCellIndices(): 
    a = M.get(x,y,z)
    M2.set(z,x,y, a)
  return M2
