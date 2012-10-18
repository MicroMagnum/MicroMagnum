set term post size 8cm, 6cm font "Helvetica" 11
set output "domainwall.eps"

Ms = 8e5
ns = 1e-9

set ylabel "<Mx>/Ms" 
set xlabel "time (ns)"
plot \
	"domainwall.odt" using ($1/ns):($3/Ms) with line lt 1 lc 1 notitle
