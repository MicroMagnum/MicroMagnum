set term post size 16cm, 8cm font "Helvetica" 11
set output "vortex-switch.ps"

ns = 1e-9
nm = 1e-9

set ylabel "vortex core pos (nm)"
set xlabel "time (ns)"
set title "Vortex core trajectory"

plot "vortex-switch.odt" using ($1/ns):($6/nm) with lines notitle, \
     "vortex-switch.odt" using ($1/ns):($7/nm) with lines notitle
