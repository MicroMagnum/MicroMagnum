#!/usr/bin/python
import magnum.magneto as m
import magnum

class LlgDiffEq(m.DiffEq):
    def diff(self,My,Mydot):
        #Mydot = My.differentiate()
        print("DIFF from python")
