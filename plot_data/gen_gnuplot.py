#!/usr/bin/python2.7

import sys

def generate_acc_plot(dataset):
    return """#!/usr/bin/env gnuplot
set terminal lua tikz
set output "#1.tex"
set xlabel "Training subset size"
set ylabel "Accuracy"
set key box inside right center
set title "\\\\textbf{#1}"
set style data lines
set style line 1 lt 1 lw 2 pt 3 lc rgb "red"
set style line 2 lt 1 lw 2 pt 3 lc rgb "green"
set style line 3 lt 1 lw 2 pt 3 lc rgb "blue"
set style line 4 lt 1 lw 2 pt 3 lc rgb "orange"
set style line 5 lt 1 lw 2 pt 3 lc rgb "magenta"
set style line 6 lt 1 lw 2 pt 3 lc rgb "black"
plot "#1.tab" every ::1 using 1:2 ls 1 title "knn",\\
     "#1.tab" every ::1 using 1:4 ls 2 title "svm",\\
     "#1.tab" every ::1 using 1:6 ls 3 title "tree",\\
     "#1.tab" every ::1 using 1:8 ls 4 title "majority",\\
     "#1.tab" every ::1 using 1:10 ls 5 title "bayes",\\
     "#1.tab" every ::1 using 1:12 ls 6 title "neural net"      
            """.replace("#1", dataset)
 
def generate_acc_err_plot(dataset):
    return """#!/usr/bin/env gnuplot
set terminal lua tikz
set output "#1_error.tex"
set xlabel "Training subset size"
set ylabel "Accuracy standard error"
set key box inside right top
set title "\\\\textbf{#1}"
set style data lines
set style line 1 lt 1 lw 2 pt 3 lc rgb "red"
set style line 2 lt 1 lw 2 pt 3 lc rgb "green"
set style line 3 lt 1 lw 2 pt 3 lc rgb "blue"
set style line 4 lt 1 lw 2 pt 3 lc rgb "orange"
set style line 5 lt 1 lw 2 pt 3 lc rgb "magenta"
set style line 6 lt 1 lw 2 pt 3 lc rgb "black"
plot "#1.tab" every ::1 using 1:3 ls 1 title "knn",\\
     "#1.tab" every ::1 using 1:5 ls 2 title "svm",\\
     "#1.tab" every ::1 using 1:7 ls 3 title "tree",\\
     "#1.tab" every ::1 using 1:9 ls 4 title "majority",\\
     "#1.tab" every ::1 using 1:11 ls 5 title "bayes",\\
     "#1.tab" every ::1 using 1:13 ls 6 title "neural net"      
            """.replace("#1", dataset)

if len(sys.argv) > 1:
    dataset = sys.argv[1]
else:
    dataset = "iris"
  
filename = '{0}.gnuplot'.format(dataset)
f = open(filename, "w")
f.write(generate_acc_plot(dataset))
f.close()

filename = '{0}_error.gnuplot'.format(dataset)
f = open(filename, "w")
f.write(generate_acc_err_plot(dataset))
f.close()
