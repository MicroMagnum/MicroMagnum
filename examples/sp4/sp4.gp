#!/usr/bin/gnuplot
set term post size 16cm, 6cm font "Helvetica" 11
set output "sp4.ps"

Ms = 8e5
ns = 1e-9

set multiplot

set size 0.5,1
set origin 0,0
set ylabel "<M>/Ms"
set xlabel "time (ns)"
set ytics 0.5
set xtics 0.2
set xrange [0 to 1];
set yrange [-1 to 1];
set title "muMAG Standard Problem #4 (a)"
plot \
	"sp4-1.odt" using ($1/ns):($3/Ms) with line lt 1 lc 1 notitle ,\
	"sp4-1.odt" using ($1/ns):($4/Ms) with line lt 1 lc 2 notitle ,\
	"sp4-1.odt" using ($1/ns):($5/Ms) with line lt 1 lc 3 notitle 

set size 0.5,1
set origin 0.5,0
set ylabel "<M>/Ms"
set xlabel "time (ns)"
set ytics 0.5
set xtics 0.2
set xrange [0 to 1];
set yrange [-1 to 1];
set title "muMAG Standard Problem #4 (b)"
plot \
	"sp4-2.odt" using ($1/ns):($3/Ms) with line lt 1 lc 1 notitle ,\
	"sp4-2.odt" using ($1/ns):($4/Ms) with line lt 1 lc 2 notitle ,\
	"sp4-2.odt" using ($1/ns):($5/Ms) with line lt 1 lc 3 notitle 
