#!/usr/bin/python
from magnum import *
from math import sin, cos, pi

Py = Material.Py(Ms=8e5, A=1.3e-11,k_uniaxial=5e2, axis1=(0,1,0))

# x: "short axis"
# y: "long axis"
world = World(
  RectangularMesh((50,100,1), (20e-9, 20e-9, 20e-9)),
  #RectangularMesh((100,200,1), (10e-9, 10e-9, 20e-9)),
  #RectangularMesh((200,400,4), (5e-9, 5e-9, 5e-9)),
  Body("square", Py, Everywhere())
)

def hysteresis(name, axis):
  f = open("hysteresis-%s-axis.txt" % name, "w+")
  f.write("# H (mT)\t<mx> <my> <mz>\n")

  solver = create_solver(
    world, [StrayField, ExchangeField, AnisotropyField, ExternalField], log=True, do_precess=False
  )
  solver.state.M = (axis[0]*Py.Ms, axis[1]*Py.Ms, axis[2]*Py.Ms)

  for H in [x*1e-3/MU0 for x in range(50,-51,-1) + range(-50,51,1)]:
    print name, H*MU0
    Hx, Hy, Hz = H*axis[0], H*axis[1], H*axis[2]
    solver.state.H_ext_offs = (Hx, Hy, Hz)
    solver.relax(1.0)
    Mx, My, Mz = solver.state.M.average()
    f.write("%s\t%s %s %s\n" % (H*MU0, Mx/Py.Ms, My/Py.Ms, Mz/Py.Ms))

    if H == 0:
      writeOMF("M_remanence-%s-%s.omf" % (name, "up" if H > H_last else "down"), solver.state.M)
    H_last = H

  f.close()

hysteresis( "long", (sin(pi/180), cos(pi/180), 0.0))
hysteresis("short", (cos(pi/180), sin(pi/180), 0.0))
