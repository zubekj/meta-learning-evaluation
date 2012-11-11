#!/usr/bin/env gnuplot
set terminal lua tikz
set output "car.tex"
set xlabel "Training subset size"
set ylabel "Accuracy"
set key box inside right center
set title "\\textbf{car}"
set style data lines
set style line 1 lt 1 lw 2 pt 3 lc rgb "red"
set style line 2 lt 1 lw 2 pt 3 lc rgb "green"
set style line 3 lt 1 lw 2 pt 3 lc rgb "blue"
set style line 4 lt 1 lw 2 pt 3 lc rgb "orange"
set style line 5 lt 1 lw 2 pt 3 lc rgb "magenta"
set style line 6 lt 1 lw 2 pt 3 lc rgb "black"
plot "car.tab" every ::1 using 1:2 ls 1 title "knn",\
     "car.tab" every ::1 using 1:4 ls 2 title "svm",\
     "car.tab" every ::1 using 1:6 ls 3 title "tree",\
     "car.tab" every ::1 using 1:8 ls 4 title "majority",\
     "car.tab" every ::1 using 1:10 ls 5 title "bayes",\
     "car.tab" every ::1 using 1:12 ls 6 title "neural net"
