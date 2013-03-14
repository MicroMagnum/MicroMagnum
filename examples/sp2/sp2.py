#!/usr/bin/python
from magnum import *
import math

s = math.sqrt(3.0)

# Material parameters
Py = Material.Py()
A = Py.A
Ms = Py.Ms
K_m = 0.5*MU0*(Ms*Ms)
l_ex = math.sqrt(A/K_m)
#print l_ex

# Geometry: ratio = d/l_ex
def geometry(ratio):
  d = ratio * l_ex
  t = 0.1 * d
  L = 5.0 * d
  return L, d, t

def discretize(L, d, t):
  nn = (10,10,10)
  dd = (1e-9,1e-9,1e-9)
  return RectangularMesh(nn, dd)

for ratio in (0.1,0.2):
  L, d, t = geometry(ratio)
  print "d/l_ex=%s, L=%s, d=%s, t=%s" % (ratio, L, d, t)

  mesh = discretize(L, d, t)
  world = World(mesh, Body("thinfilm", Material.Py(), Everywhere()))
  solver = create_solver(world, [StrayField, ExchangeField, ExternalField], log=True, do_precess=False)
  solver.state.M = (Ms,0.0,0.0)

  # do hysteresis
  H = 0.0/MU0

  while True:
    solver.state.H_ext_offs = (H/s,H/s,H/s)
    solver.relax()
    M = solver.state.M.average()

    print H*MU0, M[0]+M[1]+M[2]
    if M[0]+M[1]+M[2] == 0.0:
      H_coerc = H
      break
    H += 0.1/MU0

  while True:
    print H*MU0
    solver.state.H_ext = (H/s,H/s,H/s)
    solver.relax()
    
    if H == 0.0: 
      M_rem = solver.state.M.average()
      break
    H -= 0.1/MU0
