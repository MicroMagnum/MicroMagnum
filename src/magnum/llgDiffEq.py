#!/usr/bin/python
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

from magnum import *
import magnum.magneto as m

class LlgDiffEq(m.DiffEq):
    def __init__(self,state):
        super(LlgDiffEq, self).__init__(state.y)
        self.state=state

    def diffX(self,My,Mydot,t):
        self.state.y.assign(My)
        self.state.t = t
        self.state.flush_cache()
        Mydot.assign(self.state.differentiate())

# not used
    def diff(self,My):
        #self.state.y = My
        print("DIFF from python")
        Mydot = self.state.differentiate()
        #Mydot=My
        return Mydot

    def getY(self):
        return self.state.y

    def getState(self):
        return self.state

    def saveState(self, yout):
        self.state.y.assign(yout)
        self.state.flush_cache()

    def saveTime(self, t):
        self.state.t = t

    def substep(self):
        self.state.substep += 1
