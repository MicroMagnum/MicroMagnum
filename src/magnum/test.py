import magnum.magneto as m

class MyCvode(m.Cvode):
    #def __init__(self):
    #    super(MyCvode, self).__init__()

    def one(self, i):
        print("one from python")
        print(i)

    def f(vecf):
        print("f from python")
        return vecf
