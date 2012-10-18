set term post size 16cm, 6cm font "Helvetica" 11
set output "sp4.ps"

Ms = 8e5
ns = 1e-9

set ylabel "<M>/Ms"
set xlabel "time (ns)"
set title "muMAG Standard Problem #4 (a)"
splot "macro-spintorque.odt" using ($1/Ms):($2/Ms):($3/Ms) with line lt 1 lc 1 notitle
