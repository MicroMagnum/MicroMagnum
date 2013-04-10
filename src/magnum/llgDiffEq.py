#!/usr/bin/python
from magnum import *
import magnum.magneto as m

class LlgDiffEq(m.DiffEq):
    def __init__(self,state):
        super(LlgDiffEq, self).__init__(state.y)
        self.State=state

    #def diff(self,My,Mydot):
    def diff(self,My):
        self.State.y = My
        Mydot = self.State.differentiate()
        print("DIFF from python")
        return Mydot

    def getY(self):
        print("getY Python")
        return self.State.y
