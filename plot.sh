#! /bin/bash
gnuplot -persist <<EOF
set data style points
show timestamp
set title "$1"
set term png
set xlabel "x"
set ylabel "y"
set xrange [0:100]
set yrange [0:100]
plot "$1" index 0:0 using 1:2 title "teacher", \\
"$1" index 1:1 using 1:2 title "border",\
"$1" index 2:2 using 1:2 title "func" with lines

