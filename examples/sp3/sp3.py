#!/usr/bin/python 
from magnum import *
import math

# material parameters
K_m = 0.5*MU0*Material.Py().Ms*Material.Py().Ms
Py = Material.Py(k_uniaxial=0.1*K_m,axis1=(0,0,1),alpha=0.5)
l_ex = math.sqrt(Py.A/K_m)

def sim(n, ratio=8.52):
  def get_energies(state):
    E_stray = state.E_stray / K_m / L**3
    E_exch  = state.E_exch  / K_m / L**3
    E_aniso = state.E_aniso / K_m / L**3
    E_tot = E_stray + E_exch + E_aniso
    return E_tot, E_stray, E_exch, E_aniso, [x/Py.Ms for x in state.M.average()]
  
  def my_vortex(field, pos): 
    chirality = 1.0
    polarization = 1.0
    core_radius = 8e-9
  
    x, y, z = pos
    Mx = polarization * core_radius
    My = -(z-0.5*L) * chirality
    Mz = +(y-0.5*L) * chirality
    scale = Py.Ms / math.sqrt(Mx**2 + My**2 + Mz**2) 
    return Mx*scale, My*scale, Mz*scale

  # 8.52, 8.47, 8.52
  L = ratio*l_ex # cube side length
  #n = 16 # number of cells in each direction
  world = World(RectangularMesh((n,n,n),(L/n,L/n,L/n)), Body("cube", Py))
  solver = create_solver(world, [StrayField, ExchangeField, AnisotropyField], log=True, do_precess=False)

  solver.state.M = my_vortex
  solver.relax(2)
  writeOMF("omf/state-vortex-%s-%s.omf" % (n, ratio), solver.state.M)
  E_vortex = get_energies(solver.state)
  
  solver.state.M = (0,0,Py.Ms)
  solver.relax(2)
  writeOMF("omf/state-flower-%s-%s.omf" % (n, ratio), solver.state.M)
  E_flower = get_energies(solver.state)
  
  return E_flower, E_vortex

def full_sp3():
  rr = [r/100.0 for r in range(840, 860)]
  for n in [8,12,16,20,24,28,32,36,40,44,48]:
    for ratio in rr:
      fl, vo = sim(16, ratio)
      
      flower_E_tot, flower_E_stray, flower_E_exch, flower_E_aniso, (flower_mx,flower_my,flower_mz) = fl
      vortex_E_tot, vortex_E_stray, vortex_E_exch, vortex_E_aniso, (vortex_mx,vortex_my,vortex_mz) = vo

      f = open("log-%s.txt" % n, "a")
      f.write("ratio=%s, E_tot=(%s vs %s, diff=%s)\n" % (ratio, flower_E_tot, vortex_E_tot, abs(flower_E_tot - vortex_E_tot)))
      f.write("                                          flower: %s\n" % ((flower_E_tot, flower_E_stray, flower_E_exch, flower_E_aniso, (flower_mx,flower_my,flower_mz)),))
      f.write("                                          vortex: %s\n" % ((vortex_E_tot, vortex_E_stray, vortex_E_exch, vortex_E_aniso, (vortex_mx,vortex_my,vortex_mz)),))
      f.close()

full_sp3()
