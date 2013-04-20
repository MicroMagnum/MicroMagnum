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

import logging

import magnum.magneto as magneto
import magnum.tools as tools

# I. Custom log message formater (with colors!)

class Formatter(logging.Formatter):
    color_map = {
        logging.DEBUG: 2,
        logging.INFO: 2,
        logging.WARNING: 6,
        logging.ERROR: 1,
        logging.CRITICAL: 1
    }

    def __init__(self):
        super(Formatter, self).__init__("[%(levelname)7s] - %(message)s", "%Y-%m-%d %H:%M:%S")

    def format(self, record):
        return tools.color(Formatter.color_map[record.levelno]) + logging.Formatter.format(self, record) + tools.nocolor()

ch = logging.StreamHandler()
ch.setFormatter(Formatter())

# II. Create logger
logger = logging.getLogger("magnum")
logger.addHandler(ch)
logger.setLevel(logging.DEBUG)

info = logger.info
warning = logger.warning
debug = logger.debug
error = logger.error
critical = logger.critical
warn = logger.warn

# III. Set debug callback (called from C++ code to communicate with Python logger)
def callback(level, msg):
    if level == 0:
        logger.debug(msg)
    elif level == 1:
        logger.info(msg)
    elif level == 2:
        logger.warning(msg)
    elif level == 3:
        logger.error(msg)
    elif level == 4:
        logger.critical(msg)
magneto.setDebugCallback(callback)

# cleanup
del ch
del callback
