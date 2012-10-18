#!/usr/bin/python
from magnum import *
from frange import frange
from math import cos, sin, pi

mesh = RectangularMesh((500,250,1), (5e-9, 5e-9, 20e-9))
Py = Material.Py(k_uni=5e2, axis1=(1,0,0))
world = World(mesh, Body("square", Py, Everywhere()))

solver = create_solver(world, [StrayField, ExchangeField, AnisotropyField, ExternalField], log=True)

# Create initial state
solver.state.M = (8e5, 0, 0)

# Perform hysteresis
H_range = list(frange(+50e-3, -50e-3, -5e-3)) + list(frange(-50e-3, +50e-3, 5e-3))
for tmp in H_range:
  H = (tmp/MU0 * cos(pi/180), tmp/MU0 * sin(pi/180), 0) # in A/m
  print(H)

  solver.state.H_offs = H
  solver.relax(1.0)
