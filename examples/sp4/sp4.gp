#!/usr/bin/gnuplot
set term pdfcairo enhanced size 16cm, 6cm
set output "sp4_rk.pdf"

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
	"sp4-1_rk.odt" using ($1/ns):($3/Ms) with line lt 1 lc 1 notitle ,\
	"sp4-1_rk.odt" using ($1/ns):($4/Ms) with line lt 1 lc 2 notitle ,\
	"sp4-1_rk.odt" using ($1/ns):($5/Ms) with line lt 1 lc 3 notitle 

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
	"sp4-2_rk.odt" using ($1/ns):($3/Ms) with line lt 1 lc 1 notitle ,\
	"sp4-2_rk.odt" using ($1/ns):($4/Ms) with line lt 1 lc 2 notitle ,\
	"sp4-2_rk.odt" using ($1/ns):($5/Ms) with line lt 1 lc 3 notitle 

unset multiplot

set output "sp4_cvode.pdf"

Ms = 8e5
ns = 1e-9

set multiplot

set size 0.5,1
set origin 0,0
set ylabel "<M>/Ms"
set xlabel "time (ns)"
set ytics 0.5
set xtics 0.2
#set xrange [*:*];
set yrange [-1 to 1];
set title "muMAG Standard Problem #4 (a)"
plot \
	"sp4-1_cvode.odt" using ($1/ns):($3/Ms) with line lt 1 lc 1 notitle ,\
	"sp4-1_cvode.odt" using ($1/ns):($4/Ms) with line lt 1 lc 2 notitle ,\
	"sp4-1_cvode.odt" using ($1/ns):($5/Ms) with line lt 1 lc 3 notitle 

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
	"sp4-2_cvode.odt" using ($1/ns):($3/Ms) with line lt 1 lc 1 notitle ,\
	"sp4-2_cvode.odt" using ($1/ns):($4/Ms) with line lt 1 lc 2 notitle ,\
	"sp4-2_cvode.odt" using ($1/ns):($5/Ms) with line lt 1 lc 3 notitle 

unset multiplot
