set term post size 12cm, 12cm font "Helvetica" 11
set output "coupled-vortices-traj.ps"

ns = 1e-9
nm = 1e-9

set ylabel "displacement (nm)" 1.5,0
set xlabel "time (ns)"
set title "Vortex core displacement"
plot \
	"coupled-vortices-same-pol.odt" using ($5/ns):($6/nm) with line lt 1 lc 1 title "p1=+1, p2=+1, core pos" ,\
	"coupled-vortices-diff-pol.odt" using ($5/ns):($6/nm) with line lt 1 lc 3 title "p1=+1, p2=-1, core pos"

