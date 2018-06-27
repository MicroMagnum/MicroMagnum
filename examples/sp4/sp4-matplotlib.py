from matplotlib.pylab import *

Ms = 8e5
ns = 1e-9

def load_odt(filename):
  cols = [(0,ns),(2,Ms),(3,Ms),(4,Ms)] # [(row-id, unit)]
  data = loadtxt(filename)
  return (data[:,i]/u for i, u in cols) # t,mx,my,mz

def sp4_plot():
  subplot(2,1,1)
  t, mx, my, mz = load_odt("sp4-1.odt")
  plot(t,mx,'r',t,my,'g',t,mz,'b')
  xlabel("$t\\;(ns)$")
  ylabel("$\\langle M \\rangle / M_\\mathrm{s}$")

  subplot(2,1,2)
  t, mx, my, mz = load_odt("sp4-2.odt")
  plot(t,mx,'r',t,my,'g',t,mz,'b')
  xlabel("$t\\;(ns)$")
  ylabel("$\\langle M \\rangle / M_\\mathrm{s}$")

sp4_plot()
