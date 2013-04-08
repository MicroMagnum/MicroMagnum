#!/usr/bin/python
from magnum import *
import magnum.magneto as m

class LlgDiffEq(m.DiffEq):
    def __init__(self,state):
        super(LlgDiffEq, self).__init__()
        self.State=state

    def diff(self,My,Mydot):
        self.State.y = My
        Mydot = self.State.differentiate()
        print("DIFF from python")

    def getY(self):
        print("getY Python")
        return self.State.y
