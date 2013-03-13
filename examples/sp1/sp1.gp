#!/usr/bin/gnuplot
set term post size 16cm, 16cm font "Helvetica" 11
set output "sp1.ps"

set multiplot

set size 1,0.5
set origin 0,0
set ylabel "<M>/Ms"
set xlabel "H (mT)"
# set ytics 0.5
# set xtics 0.2
set xrange [-0.05 to +0.05];
# set yrange [-1 to 1];
set title "muMAG SP1 (long axis field)"
plot    "hysteresis-long-axis.txt" using ($1):($2) with line lt 1 lc 1 title "x", \
	"hysteresis-long-axis.txt" using ($1):($3) with line lt 1 lc 2 title "y", \
	"hysteresis-long-axis.txt" using ($1):($4) with line lt 1 lc 3 title "z"

set size 1,0.5
set origin 0,0.5
set ylabel "<M>/Ms"
set xlabel "H (mT)"
# set ytics 0.5
# set xtics 0.2
set xrange [-0.05 to +0.05];
# set yrange [-1 to 1];
set title "muMAG SP1 (short axis field)"
plot    "hysteresis-short-axis.txt" using ($1):($2) with line lt 1 lc 1 title "x", \
	"hysteresis-short-axis.txt" using ($1):($3) with line lt 1 lc 2 title "y", \
	"hysteresis-short-axis.txt" using ($1):($4) with line lt 1 lc 3 title "z"
