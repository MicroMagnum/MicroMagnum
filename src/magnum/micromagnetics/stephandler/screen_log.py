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

from .log_stephandler import LogStepHandler
import magnum.tools as tools

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

        fmt = tools.color(5) + "%s=" + tools.nocolor() + "%s";
        sep = tools.color(5) + ", "  + tools.nocolor()

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
