from magnum.meshes import VectorField
from magnum.logger import logger

import magnum.magneto as magneto

from .evolver import Evolver
from .tableaus import rkf45, cc45, dp54, rk23

class RungeKutta(Evolver):
  TABLEAUX = {
    'rkf45': rkf45, # Runge-Kutta-Fehlberg
    'cc45': cc45, # Cash-Karp
    'dp54': dp54, # Dormand-Prince
    'rk23': rk23, # Bogacki-Shampine
  }

  def __init__(self, mesh, method, stepsize_controller):
    super(RungeKutta, self).__init__(mesh)
   
    self.__tab = RungeKutta.TABLEAUX[method]()
    self.__controller = stepsize_controller

    self.__y0    =  VectorField(mesh)
    self.__y_err =  VectorField(mesh)
    self.__y_tmp =  VectorField(mesh)
    self.__k     = [VectorField(mesh) for idx in range(self.__tab.getNumSteps())]

    logger.info("Runge Kutta evolver: method is %s, step size controller is %s.", method, self.__controller)

  def evolve(self, state, t_max):
    # shortcuts
    y, y0, y_err = (state.y, self.__y0, self.__y_err)

    # This is needed to be able to roll back one step
    y0.assign(y)

    # Get time step to try.
    try:
      h_try = state.__runge_kutta_next_h
    except AttributeError:
      h_try = state.h

    while True:
      # Try a step from state.t to state.t+h_try
      dydt = self.apply(state, h_try)

      # Step size control (was h too big?)
      # calculate the minimal acceptable step size
      accept, h_new = self.__controller.adjust_stepsize(state, h_try, self.__tab.order, y, y_err, dydt)
      if not accept: # oh, tried step size was too big.
        y.assign(y0) # reverse last step
        h_try = h_new # try again with new (smaller) h.
        continue # need to retry -> redo loop
      else:
        break # done -> exit loop

    # But: Don't overshoot past t_max!
    if state.t + h_try > t_max: 
      h_try = t_max - state.t   # make h_try smaller.
      y.assign(y0)              # reverse last step
      self.apply(state, h_try)  # assume that a smaller step size is o.k.

    # Update state
    state.t += h_try
    state.h  = h_try; state.__runge_kutta_next_h = h_new
    state.step += 1
    state.substep = 0
    state.flush_cache()
    state.finish_step()
    return state

  def apply(self, state, h):
    (y, y_tmp, y_err) = (state.y, self.__y_tmp, self.__y_err)
    tab = self.__tab
    num_steps = tab.getNumSteps()

    k = [None for idx in range(num_steps)]

    # I. Calculate step vectors k[0] to k[5]

    # step 0 (if method has first-step-as-last (fsal) property, we might already know the first step vector.)
    state0 = state
    if tab.fsal and hasattr(state, "dydt_in") and state.dydt_in is not None:
      k[0] = state.dydt_in
    else:
      k[0] = state0.differentiate()

    # step 1 to 5
    for step in range(1, num_steps): 
      # calculate ytmp...
      if num_steps != 6: # High-level version for num_steps != 6
        y_tmp.assign(y)
        for j in range(0, step):
          y_tmp.add(k[j], h * tab.getB(step, j))
      else:
        # C++ version for num_steps==6 (rkf45,cc45)
        magneto.rk_prepare_step(step, h, tab, k[0], k[1] or k[0], k[2] or k[0], k[3] or k[0], k[4] or k[0], k[5] or k[0], y, y_tmp)

      state1 = state0.clone(y_tmp) 
      state1.t = state0.t + h * tab.getA(step)
      state1.substep = step
      k[step] = state1.differentiate()

    # II. Linear-combine step vectors, add them to y, calculate error y_err
    if num_steps == 3:
      magneto.rk_combine_result(h, tab, k[0], k[1], k[2], y, y_err)
    if num_steps == 6:
      magneto.rk_combine_result(h, tab, k[0], k[1], k[2], k[3], k[4], k[5], y, y_err)
    else:
      y_err.clear()
      for step in range(0, num_steps):
        y    .add(k[step], h * tab.getC (step))
        y_err.add(k[step], h * tab.getEC(step))

    # III. Exploit fsal property?
    if tab.fsal:
      # save last dydt for next step
      state.dydt_in = k[num_steps-1]

    # Also, return dydt (which is k[0])
    return k[0]
