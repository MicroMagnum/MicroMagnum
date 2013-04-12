#!/usr/bin/python
from magnum import *
import magnum.magneto as m

class LlgDiffEq(m.DiffEq):
    def __init__(self,state):
        super(LlgDiffEq, self).__init__(state.y)
        self.State=state

    def diffX(self,My,Mydot):
        print("DIFFX from python")
        Mydot=self.State.differentiate()

    def diff(self,My):
        #self.State.y = My
        print("DIFF from python")
        Mydot = self.State.differentiate()
        #Mydot=My
        return Mydot

    def getY(self):
        print("getY Python")
        return self.State.y
