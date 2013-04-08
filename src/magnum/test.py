import magnum.magneto as m

class MyCvode(m.Cvode):
    #def __init__(self,y,ydot,llg):
    #    super(MyCvode, self,y,ydot,llg).__init__()

    def one(self, i):
        print("one from python")
        print(i)

    def f(vecf):
        print("f from python")
        return vecf
