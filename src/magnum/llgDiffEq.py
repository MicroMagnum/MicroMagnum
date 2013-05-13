#!/usr/bin/python
from magnum import *
import magnum.magneto as m

class LlgDiffEq(m.DiffEq):
    def __init__(self,state):
        super(LlgDiffEq, self).__init__(state.y)
        self.State=state

    def diffX(self,My,Mydot,t):
        self.State.y.assign(My)
        self.State.t = t
        self.State.flush_cache()
        Mydot.assign(self.State.differentiate())

# not used
    def diff(self,My):
        #self.State.y = My
        print("DIFF from python")
        Mydot = self.State.differentiate()
        #Mydot=My
        return Mydot

    def getY(self):
        return self.State.y
