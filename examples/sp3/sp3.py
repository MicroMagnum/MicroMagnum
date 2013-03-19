#!/usr/bin/python 
from magnum import *
import math

K_m = 0.5*MU0*Material.Py().Ms*Material.Py().Ms
Py = Material.Py(k_uniaxial=0.1*K_m,axis1=(0,0,1),alpha=0.5)
l_ex = math.sqrt(Py.A/K_m)

def sim(N, ratio):
  def get_info(state):
    E_str = state.E_str / K_m / L**3
    E_ex  = state.E_ex  / K_m / L**3
    E_ani = state.E_ani / K_m / L**3
    E_tot = E_str + E_ex + E_ani
    return E_tot, E_st, E_ex, E_ani, [x/Py.Ms for x in state.M.average()]
  
  def my_vortex(field, pos): 
    x, y, z = pos
    Mx = 8e-9
    My = -(z-0.5*L)
    Mz = +(y-0.5*L)
    scale = Py.Ms / math.sqrt(Mx**2 + My**2 + Mz**2) 
    return Mx*scale, My*scale, Mz*scale

  L = ratio*l_ex # cube side length
  world = World(RectangularMesh((N,N,N),(L/N,L/N,L/N)), Body("cube", Py))
  solver = create_solver(world, [StrayField, ExchangeField, AnisotropyField], log=True, do_precess=False)

  solver.state.M = my_vortex
  solver.relax(2)
  writeOMF("omf/state-vortex-%s-%s.omf" % (N, ratio), solver.state.M)
  vo = get_info(solver.state)
  
  solver.state.M = (0, 0, Py.Ms)
  solver.relax(2)
  writeOMF("omf/state-flower-%s-%s.omf" % (N, ratio), solver.state.M)
  fl = get_info(solver.state)
  
  return vo, fl

def full_sp3():
  for N in [12,16,20,24,28,32,36,40,44,48]:
    for ratio in [r/100.0 for r in range(840, 860)]:
      fl, vo = sim(N, ratio)
      fl_E_tot, fl_E_str, fl_E_ex, fl_E_ani, (fl_mx, fl_my, fl_mz) = fl
      vo_E_tot, vo_E_str, vo_E_ex, vo_E_ani, (vo_mx, vo_my, vo_mz) = vo

      f = open("log-%s.txt" % N, "a")
      f.write("ratio=%s, E_tot=(%s vs %s, diff=%s)\n" % (ratio, fl_E_tot, vo_E_tot, abs(fl_E_tot - vo_E_tot)))
      f.write(" * flower: %s\n" % ((fl_E_tot, fl_E_str, fl_E_ex, fl_E_ani, (fl_mx,fl_my,fl_mz)),))
      f.write(" * vortex: %s\n" % ((vo_E_tot, vo_E_str, vo_E_ex, vo_E_ani, (vo_mx,vo_my,vo_mz)),))
      f.close()

full_sp3()
