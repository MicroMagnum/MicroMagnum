# Copyright 2012 by the Micromagnum authors.
#
# This file is part of MicroMagnum.
# 
# MicroMagnum is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# MicroMagnum is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with MicroMagnum.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import print_function

from magnum.config import cfg
from magnum.logger import logger

import sys, itertools

### Controller factory method ##############################################

def Controller(run, params, *args, **kwargs):
  ### Determine which controller class to use ###

  # Case 1: Report to stdout, then exit, if requested by command line options.
  print_num_params = cfg.options.print_num_params_requested
  print_all_params = cfg.options.print_all_params_requested
  if print_num_params or print_all_params:
    class PrintParametersController(ControllerBase):
      def __init__(self, run, params, *args, **kwargs):
        ControllerBase.__init__(self, run, params)
      def start(self):
        if print_num_params:
          print("NUM_PARAMETERS %s" % len(self.getAllParameters()))
        if print_all_params:
          for idx in range(len(self.getAllParameters())):
            param = self.getAllParameters()[idx]
            print("PARAMETER %s %s" % (idx, param))
    cont = PrintParametersController

  # Case 2: This controller is used when the script was executed locally
  elif True:
    from .localcontroller import LocalController
    cont = LocalController

  # Error case: Bail out if not quitable controller was found (can not happen!?)
  else:
    raise RuntimeError("Couldn't agree on a controller class to use.")

  ### Create controller object and return ###
  return cont(run, params, *args, **kwargs)

### Controller base class ##################################################

class ControllerBase(object):
  def __init__(self, run, params):
    # Run function
    if not hasattr(run, '__call__'):
      raise TypeError("LocalController: 'run' argument must be callable")
    self.__run = run
    # Parameter set
    self.__all_params = _unpackParameters(params)

  def getRunFunction(self):
    return self.__run

  def getAllParameters(self):
    return self.__all_params

  def start(self):
    raise NotImplementedError("ControllerBase: start needs to be implemented in sub-class")

  def logCallMessage(self, idx, param):
    if len(param) == 1: (p,) = param
    else: p = param
    logger.info("==========================================================")
    logger.info("Controller: Calling simulation function (param set: %s)", idx)
    logger.info("            Parameters: %s", p)

### Convert parameter specs to list of parameter tuples #######################

def _unpackParameters(param_spec):
  # I. Preprocess tuples
  def mapping(p):
    def scalar_to_list(i):
      if type(i) == list: return i
      else: return [i]
    # treat scalars as 1-tuples
    if not type(p) == tuple: p = (p,)
    # map scalar tuple entries to lists (with one scalar entry)
    return tuple(map(scalar_to_list, p))
  param_spec = list(map(mapping, param_spec))

  # II. Map parameters to their cartesian product
  result = []
  for param in param_spec:
    for element in itertools.product(*param):
      result.append(element)

  return result
