#!/usr/bin/python
from magnum import *
import magnum.magneto as m

class LlgDiffEq(m.DiffEq):
    def __init__(self,state):
        super(LlgDiffEq, self).__init__(state.y)
        self.State=state

    def diffX(self,My,Mydot,t):
        print("DIFFX from python")
        self.State.y.assign(My)
        print("DIFFX 1")
        self.State.t = t
        print("DIFFX 2")
        self.State.flush_cache()
        print("DIFFX 3")
        Mydot.assign(self.State.differentiate())
        print("DIFFX 4")

    def diff(self,My):
        #self.State.y = My
        print("DIFF from python")
        Mydot = self.State.differentiate()
        #Mydot=My
        return Mydot

    def getY(self):
        print("getY Python")
        return self.State.y
