# Copyright 2012, 2013 by the Micromagnum authors.
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

import os
import itertools

import magnum.logger as logger
from magnum.config import cfg

# Controllers defined here:
#   LocalController, PrintParametersController,
#   EnvironmentVariableController, SunGridEngineController


class ControllerBase(object):
    """
    Base class for all controllers.
    """

    def __init__(self, run, params):

        if not hasattr(run, '__call__'):
            raise TypeError("Controller: 'run' argument must be callable")

        self.run = run
        self.all_params = list(enumerate(unpackParameters(params)))
        self.num_params = len(self.all_params)

    def logCallMessage(self, idx, param):
        if len(param) == 1:
            param, = param
        logger.info("==========================================================")
        logger.info("Controller: Calling simulation function (param set: %s)", idx)
        logger.info("            Parameters: %s", param)

    def start(self):
        for idx, param in self.my_params:
            self.logCallMessage(idx, param)
            self.run(*param)

    def select(self, range):
        self.my_params = [self.all_params[i] for i in range if 0 <= i < self.num_params]


def unpackParameters(param_spec):
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


class LocalController(ControllerBase):
    """
    This controller will iterate through all parameter sets, if no -p
    argument is given. If -p=a,b is given, the controller will iterate
    through the interval given by range(a,b), i.e. from a to (b-1) inclusive.
    """

    def __init__(self, run, params):
        super(LocalController, self).__init__(run, params)

        idx_range = getattr(
            cfg.options, 'prange',
            range(self.num_params)
        )

        for i in idx_range:
            if 0 <= i < self.num_params:
                continue
            logger.warn("Controller: No such parameter set with index %s!" % i)

        if len(idx_range) == 0:
            logger.warn("Controller: No parameter sets selected!")

        self.select(idx_range)


class PrintParametersController(ControllerBase):
    """
    """

    def __init__(self, run, params, print_num_params, print_all_params):
        super(PrintParametersController, self).__init__(run, params)

        self.print_num_params = print_num_params
        self.print_all_params = print_all_params
        self.select([])

    def start(self):
        if self.print_num_params:
            print("NUM_PARAMETERS %s" % self.num_params)
        if self.print_all_params:
            for param in self.all_params:
                print("PARAMETER %s %s" % (param[0], param[1]))


class EnvironmentVariableController(ControllerBase):
    """
    This controller uses an environment variable to select exactly one
    parameter set.
    """

    def __init__(self, run, params, env, offset=0):
        super(EnvironmentVariableController, self).__init__(run, params)

        try:
            p_idx = int(os.environ[env]) - offset
        except:
            logger.error("Could not read environment variable '%s'." % env)
            raise

        if 0 >= p_idx < self.num_params:
            self.select([p_idx])
        else:
            logger.warn("Controller: No such parameter set with index %s!" % p_idx)
            self.select([])


class SunGridEngineController(EnvironmentVariableController):
    """
    This controller uses the 'SGE_TASK_ID' enviroment variable to select
    one parameter set. To be used in conjunction with task arrays using
    the Sun Grid Engine.
    """

    def __init__(self, run, params):
        super(SunGridEngineController, self).__init__(
            run, params,
            env="SGE_TASK_ID", offset=1,
        )
