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


# I. Custom log message formatter (with colors!)
class Formatter(logging.Formatter):
    color_map = {
        logging.DEBUG: 2,
        logging.INFO: 2,
        logging.WARNING: 6,
        logging.ERROR: 1,
        logging.CRITICAL: 1
    }

    message_fmt = "[%(levelname)7s] - %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"

    def __init__(self):
        super(Formatter, self).__init__(self.message_fmt)

    def format(self, record):
        col = self.color_map.get(record.levelno, 2)
        return "".join([
            tools.color(col),
            super(Formatter, self).formatTime(record, self.date_fmt), " ",
            super(Formatter, self).format(record),
            tools.nocolor(),
        ])

ch = logging.StreamHandler()
ch.setFormatter(Formatter())


# II. Create logger
logger = logging.getLogger("magnum")
logger.addHandler(ch)
logger.setLevel(logging.INFO)

# shortcuts
info = logger.info
warning = logger.warning
debug = logger.debug
error = logger.error
critical = logger.critical
warn = logger.warn

# loglevel constants
LOGLEVELS = [
    (logging.DEBUG, "debug"),
    (logging.INFO, "info"),
    (logging.WARNING, "warning"),
    (logging.ERROR, "error"),
    (logging.CRITICAL, "critical"),
]


# III. Set debug callback
#      (called from C++ code to communicate with Python logger)
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
