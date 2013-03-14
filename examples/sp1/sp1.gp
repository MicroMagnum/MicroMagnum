#!/usr/bin/gnuplot
set term post size 16cm, 16cm font "Helvetica" 11
set output "sp1.ps"

set multiplot

set size 1,0.5
set origin 0,0
set ylabel "<M>/Ms"
set xlabel "mu0 H (mT)"
set ytics 0.5
set xrange [-50 to +50];
set xtics 25
set yrange [-1 to 1];
set title "muMAG SP1 (short axis field)"
plot    "hysteresis-short-axis.txt" using ($1/1e-3):($2) with line lt 1 lc 1 title "x", \
	"hysteresis-short-axis.txt" using ($1/1e-3):($3) with line lt 1 lc 2 title "y"

set size 1,0.5
set origin 0,0.5
set ylabel "<M>/Ms"
set xlabel "mu0 H (mT)"
set ytics 0.5
set xrange [-50 to +50];
set xtics 25
set yrange [-1 to 1];
set title "muMAG SP1 (long axis field)"
plot    "hysteresis-long-axis.txt" using ($1/1e-3):($2) with line lt 1 lc 1 title "x", \
	"hysteresis-long-axis.txt" using ($1/1e-3):($3) with line lt 1 lc 2 title "y"
