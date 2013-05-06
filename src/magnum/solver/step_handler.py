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

class StepHandler(object):
    def __init__(self):
        pass

    def handle(self, state):
        """
        This method is called by the solver when a new simulation step
        needs to be processed by the step handler.  It must be overridden
        by a sub-class.
        """
        raise NotImplementedError("StepHandler.handle(): Implement me!")

    def done(self):
        """
        This method is called by the solver when a simulation is complete
        (before the ''solver'' method returns). Step handlers can use this
        method to clean up, close log files etc.
        """
        pass
