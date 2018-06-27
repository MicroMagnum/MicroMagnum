#!/usr/bin/python
from magnum import *
from math import *

def find_bug(n):
  if (n == 0): mesh = RectangularMesh((1, 200, 30), (1e-9, 1e-9, 1e-9))   # endlosschleife
  if (n == 1): mesh = RectangularMesh((200, 30, 1), (1e-9, 1e-9, 1e-9))   # zerodivision
  if (n == 2): mesh = RectangularMesh((200, 1, 30), (1e-9, 1e-9, 1e-9))   # ok

  world = World(mesh, Body('film', Material.Py()))
  solver = create_solver(world, [StrayField], log=True, evolver="euler")
  solver.state.M = (0,0,8e5)
  solver.solve(condition.Time(1e-9))

params = [0, 1, 2]
c = Controller(find_bug, params)
c.start()
