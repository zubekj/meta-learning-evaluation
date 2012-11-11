#!/usr/bin/env gnuplot
set terminal lua tikz
set output "glass.tex"
set xlabel "Training subset size"
set ylabel "Accuracy"
set key box inside right center
set title "\\textbf{glass}"
set style data lines
set style line 1 lt 1 lw 2 pt 3 lc rgb "red"
set style line 2 lt 1 lw 2 pt 3 lc rgb "green"
set style line 3 lt 1 lw 2 pt 3 lc rgb "blue"
set style line 4 lt 1 lw 2 pt 3 lc rgb "orange"
set style line 5 lt 1 lw 2 pt 3 lc rgb "magenta"
set style line 6 lt 1 lw 2 pt 3 lc rgb "black"
plot "glass.tab" every ::1 using 1:2 ls 1 title "knn",\
     "glass.tab" every ::1 using 1:4 ls 2 title "svm",\
     "glass.tab" every ::1 using 1:6 ls 3 title "tree",\
     "glass.tab" every ::1 using 1:8 ls 4 title "majority",\
     "glass.tab" every ::1 using 1:10 ls 5 title "bayes",\
     "glass.tab" every ::1 using 1:12 ls 6 title "neural net"
