#!/usr/bin/gnuplot
set term post size 16cm, 6cm font "Helvetica" 11
set output "sp0.ps"

f(x) = A * sin(B*x + C)
A = 600000;
B = 3.52e10*3.1416*2;
C = 0.5;

fit f(x) "larmor.odt" using 1:3 via A, B, C

plot "larmor.odt" using ($1):($3) with line lt 1 lc 1 notitle, f(x)
