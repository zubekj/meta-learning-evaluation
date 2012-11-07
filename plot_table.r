args = commandArgs(TRUE)
t <- read.table(args[1], header=TRUE)

min_acc = min(t[,-1])
max_acc = max(t[,-1])

plot_colors <- c("blue","red","forestgreen","orange","magenta","black")

plot(t$Data_subset, t$knn, col=plot_colors[1], ylim=c(min_acc,max_acc),
     type="l", ann = FALSE)
lines(t$Data_subset, t$svm, col=plot_colors[2])
lines(t$Data_subset, t$tree, col=plot_colors[3])
lines(t$Data_subset, t$majority, col=plot_colors[4])
lines(t$Data_subset, t$bayes, col=plot_colors[5])
lines(t$Data_subset, t$neural_net, col=plot_colors[6])

title(xlab="Data subset size")
title(ylab="Accuracy")
legend(0.1, max_acc, names(t[,-1]), cex=0.8, col=plot_colors, lty=1)

split_path = unlist(strsplit(args[1], "/"))
split_filename = unlist(strsplit(split_path[length(split_path)], "\\."))
title(main=split_filename[[1]])
