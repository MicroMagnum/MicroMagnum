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

from magnum.solver import StepHandler

import sys, time

class LogStepHandler(StepHandler):

    class MultiColumn(object):
        class Column(object):
            def __init__(self, id, desc, unit, fmt):
                self.id = id
                self.desc = desc
                self.fmt = fmt
                self.unit = unit

        def __init__(self, fn):
            self.columns = []
            self.func = fn
        def addColumn(self, id, desc = "(no descr.)", unit = "unknown", fmt = "%r"):
            self.columns.append(LogStepHandler.MultiColumn.Column(id, desc, unit, fmt))
        def getColumn(self, n):
            return self.columns[n]
        def count(self):
            return len(self.columns)

    def __init__(self, out_stream):
        super(LogStepHandler, self).__init__()

        #assert type(out_stream) == file, "out_stream parameter must be of type 'file'"

        self.f = out_stream
        self.first_call = True
        self.columns = []
        self.__walltime0 = time.time()

    def writeHeader(self):
        row = "# "
        for mc in self.columns:
            for col in mc.columns:
                row += "%s: %s (%s)   " % (col.desc, col.id, col.unit)
        self.f.write(row + "\n")

    def writeAppendix(self):
        self.f.write("# done\n")

    def handle(self, state):
        # Write header at first call
        if self.first_call:
            self.writeHeader()
            self.first_call = False

        # Write table row entry
        row = self.generateRow(state)

        self.f.write(row + "\n")
        self.f.flush()

    def generateRow(self, state):
        raise NotImplementedError("LogStepHandler.generateRow is abstract")

    def done(self):
        self.writeAppendix()

    def addColumn(self, *args):
        """
        Adds a new column to the log. TODO: Examples
        """
        if not len(args) >= 2: raise TypeError("Not enough arguments.")
        func     = args[-1]
        col_spec = args[0:-1]
        if not hasattr(func, '__call__'): raise TypeError("Last argument must be a callable function")

        mc = LogStepHandler.MultiColumn(func)
        for spec in col_spec:
            if not isinstance(spec, tuple): spec = (spec,)
            mc.addColumn(*spec)
        self.columns.append(mc)

    def addTimeColumn(self):
        """
        Adds a column with the simulation time to the log.
        """
        self.addColumn(("t", "time", "s", "%r"), lambda state: state.t)

    def addWallTimeColumn(self):
        def walltime(state):
            t0 = self.__walltime0 # at begin of simulation
            t1 = time.time()      # now
            return t1 - t0
        self.addColumn(("t_wall", "wall clock time", "s", "%.3f"), walltime)

    def addStepSizeColumn(self):
        """
        Adds a column with the step size to the log.
        """
        self.addColumn(("h", "step size", "s", "%r"), lambda state: state.h)

    def addAverageMagColumn(self):
        """
        Adds a column with the spatially averaged magnetization to the log.
        """
        self.addColumn(
          ("Mx", "average magn. x", "A/m", "%r"),
          ("My", "average magn. y", "A/m", "%r"),
          ("Mz", "average magn. z", "A/m", "%r"),
          lambda state: state.M.average()
        )

    def addEnergyColumn(self, energy_variable):
        """
        Adds a column with the field energy interals.
        Examples:
          sh.addEnergyColumn("E_stray")
          sh.addEnergyColumn("E_exch")
          sh.addEnergyColumn("E_tot")
        """
        self.addColumn((energy_variable, "%s-energy" % energy_variable, "J", "%r"), lambda state: getattr(state, energy_variable))
