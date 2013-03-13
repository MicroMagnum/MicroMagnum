#!/usr/bin/python
from magnum import *
from frange import frange
from math import cos, sin, pi

#mesh = RectangularMesh((50,100,1), (20e-9, 20e-9, 20e-9))
mesh = RectangularMesh((25,50,1), (40e-9, 40e-9, 20e-9))
Py = Material.Py(k_uniaxial=5e2, axis1=(1,0,0))
world = World(mesh, Body("square", Py, Everywhere()))

solver = create_solver(world, [StrayField, ExchangeField, AnisotropyField, ExternalField], log=True)
solver.state.M = (8e5, 0, 0)

# Perform hysteresis
H_range = list(frange(+50e-3, -50e-3, -5e-3)) + list(frange(-50e-3, +50e-3, 5e-3))
H_range = [H/MU0 for H in H_range] # convert Tesla->A/m

def hysteresis(axis):
  if axis == "long":
    ax = (sin(pi/180), cos(pi/180), 0.0) # along 'long' y axis
  elif axis == "short":
    ax = (cos(pi/180), sin(pi/180), 0.0) # along 'short' x axis
  else:
    raise ValueError("need to specify 'long' or 'short'")

  f = open("hysteresis-%s-axis.txt" % axis, "w+")
  f.write("# H (mT)\tmx(A/m) my(A/m) mz(A/m)\n")
  for H in H_range:
    print axis, H
  
    Hx, Hy, Hz = H*ax[0], H*ax[1], H*ax[2]
    solver.state.H_ext_offs = (Hx, Hy, Hz)
    solver.relax(1.0)
    mx, my, mz = [a/Py.Ms for a in solver.state.M.average()]
    
    f.write("%s\t%s %s %s\n" % (H*MU0, mx, my, mz))
    f.flush()
  f.close()

hysteresis("long")
hysteresis("short")
