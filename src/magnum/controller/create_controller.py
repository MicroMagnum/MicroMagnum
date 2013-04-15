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

from magnum.config import cfg
from magnum.logger import logger

import os

from .controller import LocalController, EnvironmentVariableController, SunGridEngineController, PrintParametersController

def create_controller(run, params, *args, **kwargs):
    """
    Create a controller object, depending on the environment in which the script was executed:
    TODO: Explain.
    """

    # Case 1: Report to stdout, then exit, if requested by command line options.
    print_num_params = cfg.options.print_num_params_requested
    print_all_params = cfg.options.print_all_params_requested
    if print_num_params or print_all_params:
        return PrintParametersController(run, params, print_num_params, print_all_params, *args, **kwargs)

    # Case 2: Use environment variable to select parameter set
    env = kwargs.pop("env", None)
    env_offset = kwargs.pop("env_offset", 0)
    if env:
        if env in os.environ:
            return EnvironmentVariableController(run, params, env, env_offset, *args, **kwargs)
        else:
            logger.warn("Environment variable '%s' not found, ignoring 'env' parameter in controller creation" % env)

    # Case 3: Sun grid engine controller
    sge = kwargs.pop("sun_grid_engine", False)
    if sge:
        if "SGE_TASK_ID" in os.environ:
            return SunGridEngineController(run, params, *args, **kwargs)
        else:
            logger.warn("Environment variable 'SGE_TASK_ID' not found, ignoring 'sun_grid_engine' parameter in controller creation")

    # Case 4: This controller is used when the script was executed locally. It optionally uses the -p argument passed to the sim script.
    return LocalController(run, params, *args, **kwargs)

def Controller(run, params, *args, **kwargs):
    logger.warn("The 'Controller' function is deprecated, please use 'create_controller' instead.")
    return create_controller(run, params, *args, **kwargs)
