from .log_stephandler import LogStepHandler
import magnum.console as console

import sys

class ScreenLog(LogStepHandler):
  """
  This step handler produces a log of the simulations on the screen. By
  default, the simulation time, the step size, and the averaged
  magnetizations is included in the log.
  """

  def __init__(self):
    super(ScreenLog, self).__init__(sys.stdout)
    self.addTimeColumn()
    self.addStepSizeColumn()
    self.addAverageMagColumn()
    self.addWallTimeColumn()
    self.addColumn(("deg_per_ns", "deg_per_ns", "deg/ns", "%r"), lambda state: state.deg_per_ns)

  def generateRow(self, state): 
    first = True

    fmt = console.color(5) + "%s=" + console.nocolor() + "%s";
    sep = console.color(5) + ", "  + console.nocolor()

    row = ""
    for mc in self.columns:
      values = mc.func(state)
      if type(values) != tuple: values = (values,)
      for n, col in enumerate(mc.columns):
        if not first: 
          row += sep
        else: 
          first = False
        row += fmt % (col.id, col.fmt % values[n])
    return row

  def done(self):
    LogStepHandler.done(self)
  
