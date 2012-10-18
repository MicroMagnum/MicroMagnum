set term post size 16cm, 6cm font "Helvetica" 11
set output "coupled-vortices.ps"

ns = 1e-9
nm = 1e-9

set ylabel "displacement (nm)"
set xlabel "time (ns)"
set title "Vortex core displacement"
plot \
	"coupled-vortices-same-pol.odt" using ($1/ns):($7/nm) with line lt 1 lc 1 title "p1=1, p2=1, core X" ,\
	"coupled-vortices-same-pol.odt" using ($1/ns):($8/nm) with line lt 1 lc 2 title "p1=1, p2=1, core Y" ,\
	"coupled-vortices-diff-pol.odt" using ($1/ns):($9/nm) with line lt 1 lc 3 title "p1=1, p2=-1, core X" ,\
	"coupled-vortices-diff-pol.odt" using ($1/ns):($10/nm) with line lt 1 lc 4 title "p1=1, p2=-1, core Y" 

