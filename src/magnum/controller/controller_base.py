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

from magnum.logger import logger

import itertools

class ControllerBase(object):

    def __init__(self, run, params, *args, **kwargs):
        if not hasattr(run, '__call__'):
            raise TypeError("Controller: 'run' argument must be callable")
        self.__run = run
        self.__all_params = ControllerBase.unpackParameters(params)

    run        = property(lambda self: self.__run)
    all_params = property(lambda self: self.__all_params)
    num_params = property(lambda self: len(self.__all_params))

    def start(self):
        raise NotImplementedError("ControllerBase: start needs to be implemented in sub-class")

    def logCallMessage(self, idx, param):
        if len(param) == 1: (p,) = param
        else: p = param
        logger.info("==========================================================")
        logger.info("Controller: Calling simulation function (param set: %s)", idx)
        logger.info("            Parameters: %s", p)

    @staticmethod
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
