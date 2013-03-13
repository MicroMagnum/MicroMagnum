#!/usr/bin/python 
from magnum import *
import math

s = math.sqrt(3.0)

# material parameters
K_m = 0.5*MU0*Material.Py().Ms*Material.Py().Ms
Py = Material.Py(k_uniaxial=0.1*K_m,axis1=(0,0,1),alpha=0.5)
l_ex = math.sqrt(Py.A/K_m)

# 8.47, 8.52
L = 8.52*l_ex # cube side length
n = 16 # number of cells in each direction
world = World(RectangularMesh((n,n,n),(L/n,L/n,L/n)), Body("cube", Py))

solver = create_solver(world, [StrayField, ExchangeField, AnisotropyField], log=True, do_precess=False)

def get_energies(state):
  E_stray = state.E_stray / K_m / L**3
  E_exch  = state.E_exch  / K_m / L**3
  E_aniso = state.E_aniso / K_m / L**3
  E_tot = E_stray + E_exch + E_aniso
  return E_tot, [E_stray, E_exch, E_aniso], [x/Py.Ms for x in state.M.average()]

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

solver.state.M = my_vortex
solver.relax(2)
writeOMF("state-vortex.omf", solver.state.M)
E_vortex = ("vortex",) + get_energies(solver.state)

solver.state.M = (0,0,Py.Ms)
solver.relax(2)
writeOMF("state-flower.omf", solver.state.M)
E_flower = ("flower",) + get_energies(solver.state)

print E_flower
print E_vortex
