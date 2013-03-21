#!/usr/bin/gnuplot
set term post size 16cm, 16cm font "Helvetica" 11
set output "sp2.ps"

set multiplot

set size 1,0.5
set origin 0,0.5
set ylabel "<Mx>/Ms"
set xlabel "d/l_ex"
set xrange [0 to 40];
set yrange [0.95 to 1];
set title "muMAG Standard Problem #2"
plot "log.txt" using ($1):($2) with points notitle

set size 1,0.5
set origin 0,0
set ylabel "<My>/Ms"
set xlabel "d/l_ex"
set xrange [0 to 40];
set yrange [0 to 0.1];
set title " "
plot "log.txt" using ($1):($3) with points notitle
