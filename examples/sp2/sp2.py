#!/usr/bin/python
from magnum import *
import math

# material parameters
Py = Material.Py()
Ms = Py.Ms
K_m = 0.5*MU0*(Ms*Ms)
l_ex = math.sqrt(Py.A/K_m)

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
  assert dx <= l_ex and dy <= l_ex and dz <= l_ex
  return RectangularMesh((nx, ny, nz), (dx, dy, dz))

def field(A, axis = (1,1,1)):
  return tuple(A * axis[i] / math.sqrt(axis[0]**2 + axis[1]**2 + axis[2]**2) for i in (0,1,2))

for ratio in range(4,40+1):
  L, d, t = geometry(ratio)
  print "d/l_ex=%s, L=%s, d=%s, t=%s" % (ratio, L, d, t)

  mesh = discretize(L, d, t)
  world = World(mesh, Body("thinfilm", Py, Everywhere()))
  solver = create_solver(world, [StrayField, ExchangeField, ExternalField], log=False, do_precess=False)
  solver.state.M = (Ms,0.0,0.0)

  # 1. do hysteresis
  H = 10.0e-3/MU0
  while True:
    solver.state.H_ext_offs = field(H, (1,1,1))
    solver.relax(1.0)

    h = H/Ms
    m = tuple(a/Ms for a in solver.state.M.average())
    print h, m, sum(m)*Ms

    # remember remanence mag
    if abs(h) < 1e-10:
      m_rem = m

    # find coercitivity
    if h < 0 and sum(m) < 1000/Ms:
      h_coerc = abs(h)
      break # exit hysteresis loop

    # next hysteresis step
    H -= 0.1e-3/MU0

  # 2. log results
  f = open("sp2.dat", "a")
  f.write("%s    %s    %s %s %s\n" % (ratio, h_coerc, m_rem[0], m_rem[1], m_rem[2]))
  f.close()
