#Plot predcitions


#Change the name of csv file according your ML results

tbl <- read.csv("tenessen10btlnkcs.csv", header = TRUE)
a <- data.frame(tbl)
a <- a[,-1]


st.names <- c("Neutral", "Soft", "Hard", "LinkedSoft", "LinkedHard")

#Change the x.val according to number of bottleneck in a population
x.val <- seq(1, 10)

matplot(
  x.val,
  a,
  bty = "n",
  type = "l",
  xlab = "Populations",
  ylab = "Frequency",
  main = "Predictions, Tenessen 10 Btlncks (indp.)",
  lwd = 2,
  lty = 1,
  pch = 16,
  ylim = c(0,1),
  #axes = FALSE,
  xaxt ='n',
  col = rep(c("blue","chartreuse4","red2", "chartreuse1", "tomato"), length(st.names)))

legend("topright", st.names, pch = 16, col = c("blue","chartreuse4","red2", "chartreuse1", "tomato"), cex = 0.8)
#axis(1,at = seq(1,20, by = 1), font = 0.1)
axis(1,at = seq(1,10, by = 1), cex.axis = 0.7)
axis(2, at = seq(0 ,1,by = 0.1))

