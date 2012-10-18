from magnum.logger import logger
from magnum.config import cfg

from .controller import ControllerBase

### Local controller class #################################################

#
# This controller will iterate through all parameter sets, if no --prange
# argument is given. If --prange=a,b is given, the controller will iterate
# through the interval given by range(a,b).
#

class LocalController(ControllerBase):
  def __init__(self, run, params, **options):
    ControllerBase.__init__(self, run, params)

    if hasattr(cfg.options, 'prange'):
      prange = cfg.options.prange
    else:
      prange = range(len(self.getAllParameters()))

    self.__prange = []
    for i in prange:
      if i < len(self.getAllParameters()):
        self.__prange.append(i)
      else:
        logger.warn("Ignoring parameter id %s (no such parameter set!)" % i)

    if len(self.__prange) == 0:
      logger.warn("Controller: No parameter sets selected!")

  def getMyParameterRange(self):
    return self.__prange

  def getMyParameters(self):     
    return [self.getAllParameters()[idx] for idx in self.getMyParameterRange()]

  def start(self):
    func = self.getRunFunction()
    for idx in self.getMyParameterRange():
      param = self.getAllParameters()[idx]
      self.logCallMessage(idx, param)
      func(*param)
