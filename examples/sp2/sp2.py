#!/usr/bin/python
from magnum import *
import math

# Material parameters
Py = Material.Py()
A = Py.A
Ms = Py.Ms
K_m = 0.5*MU0*(Ms*Ms)
l_ex = math.sqrt(A/K_m)

def geometry(ratio):
  d = ratio * l_ex
  t = 0.1 * d
  L = 5.0 * d
  return L, d, t

def discretize(L, d, t):
  def choose(l):
    n = 1+int(1.0*l / l_ex)
    return n, l / float(n)
  nx, dx = choose(L)
  ny, dy = choose(d)
  nz, dz = choose(t)
  return RectangularMesh((nx, ny, nz), (dx, dy, dz))

def field(A, axis = (1,1,1)):
  return tuple(A * axis[i] / (axis[0]**2 + axis[1]**2 + axis[2]**2) for i in (0,1,2))

for ratio in range(1,40+1):
  L, d, t = geometry(ratio)
  print "d/l_ex=%s, L=%s, d=%s, t=%s" % (ratio, L, d, t)

  mesh = discretize(L, d, t)
  print "nn:", mesh.num_nodes
  print "cells:", mesh.delta
  
  world = World(mesh, Body("thinfilm", Py, Everywhere()))
  solver = create_solver(world, [StrayField, ExchangeField, ExternalField], log=False, do_precess=False)
  solver.state.M = (Ms,0.0,0.0)

  # do hysteresis
  H = 10.0e-3/MU0
  while True:
    solver.state.H_ext_offs = field(H, (1,1,1))
    solver.relax(1.0)

    h = H/Py.Ms
    m = tuple(a/Py.Ms for a in solver.state.M.average())
    
    print h, m
    if abs(H) < 1e-10:
      f = open("log.txt", "a")
      f.write("%s    %s %s %s\n" % (ratio, m[0], m[1], m[2]))
      f.close()
      break

    H -= 0.1e-3/MU0
