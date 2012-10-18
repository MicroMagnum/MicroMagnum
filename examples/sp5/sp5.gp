#!/usr/bin/gnuplot
set term post size 32cm, 16cm font "Helvetica" 11
set output "sp5.ps"

Ms = 8e5
ns = 1e-9
nm = 1e-9

set multiplot

set size 0.5,1
set origin 0,0
set xlabel "Vortex core pos X (nm)"
set ylabel "Vortex core pos Y (nm)"
set title "Spin torque standard Problem"
plot "sp5.odt" using ($7/nm):($8/nm) with line lt 1 lc 1 notitle;

set size 0.5, 0.5
set origin 0.5,0.5
set xlabel "time (ns)"
set ylabel "Vortex core pos X (nm)"
set title "Spin torque standard Problem"
plot "sp5.odt" using ($1/ns):($7/nm) with line lt 1 lc 1 notitle;

set size 0.5, 0.5
set origin 0.5,0.0
set xlabel "time (ns)"
set ylabel "Vortex core pos Y (nm)"
set title "Spin torque standard Problem"
plot "sp5.odt" using ($1/ns):($7/nm) with line lt 1 lc 1 notitle;
