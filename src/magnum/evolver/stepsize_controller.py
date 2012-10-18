import magnum.magneto as magneto

class StepSizeController(object):
  def __init__(self):
    pass

  def adjust_stepsize(self, state, h, order, y, dydt, y_err):
    raise NotImplementedError("StepSizeController.adjust_stepsize")
    #return accept, new_h

# Algorithm from Numerical Recepies (NR) book
class NRStepSizeController(StepSizeController):
  def __init__(self, eps_abs = 1e-3, eps_rel = 1e-3):
    self.eps_abs = eps_abs
    self.eps_rel = eps_rel

  def adjust_stepsize(self, state, h, order, y, y_err, dydt):
    h_new = magneto.rk_adjust_stepsize(order, h, self.eps_abs, self.eps_rel, y, y_err)
    accept = (h_new <= h)
    return accept, h_new

  def __str__(self):
    return "NR(eps_abs=%s, eps_rel=%s)" % (self.eps_abs, self.eps_rel)

class FixedStepSizeController(StepSizeController):
  def __init__(self, h):
    self.h = h

  def adjust_stepsize(self, state, h, order, y, y_err, dydt):
    return True, self.h
