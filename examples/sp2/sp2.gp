#!/usr/bin/gnuplot
set term post size 16cm, 24 font "Helvetica" 11
set output "sp2.ps"

set multiplot

set size 1,0.33
set origin 0,0.66
set ylabel "<Mx>/M_s"
set xlabel "d/l_ex"
set xrange [0 to 40];
set yrange [0.95 to 1];
set title "muMAG Standard Problem #2"
plot "sp2.dat" using ($1):($3) with points notitle

set size 1,0.33
set origin 0,0.33
set ylabel "<My>/M_s"
set xlabel "d/l_ex"
set xrange [0 to 40];
set yrange [0 to 0.1];
set title " "
plot "sp2.dat" using ($1):($4) with points notitle

set size 1,0.33
set origin 0,0.0
set ylabel "H_c/M_s"
set xlabel "d/l_ex"
set xrange [0 to 40];
set yrange [0.04 to 0.07];
set title " "
plot "sp2.dat" using ($1):($2) with points notitle
