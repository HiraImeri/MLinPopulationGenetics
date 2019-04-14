pred10 <- readLines("bot20enlargetog_20.out")
pred10ar <- as.numeric(pred10)
new.df <- data.frame(pred = pred10ar)
dn <- read.table("bot20enlarge_20.fvec", header = TRUE)
d <- cbind(dn,new.df)
par(mfrow=c(2,3),oma=c(0,0,0,0))


stats10 <- function(d,i){
  n.win <- 11
  win.length <- 110000 / n.win
  newdata <- subset(d, pred == i, select = pi_win0:ZnS_win10)
  st.names <- c("Pi", "Segregating Sites", "Theta H", "Tajima's D", "Fay and Wu's H",
                "Haplotype Count", "H1", "H12", "H2/H1", "Omega", "Zns")
  means <- apply(newdata, 2, mean)
  mat <- matrix(NA, ncol = n.win, nrow = length(st.names))
  for(st.to.plot in 1:length(st.names)) {
    start <- (st.to.plot - 1) * n.win + 1
    end <- start + n.win - 1
    col.to.sel <- start:end
    mat[st.to.plot, ] <- means[col.to.sel]}
  x.val <- seq(0, win.length, length.out = n.win)
  matplot(
    x.val,
    t(mat),
    bty = "n",
    type = "l",
    xlab = "Position (bp)",
    ylab = "Relative value of statistic",
    cex.main = 0.8,
    main = "LinkedHard sweep",
    ylim = c(0.0,0.2),
    lwd = 2,
    lty = 1,
    col = rep(rainbow(12), length(st.names))
  )
  legend("topright", st.names, lty = 1, col = rainbow(12), cex = 0.3)
}


stats10(d,0)
stats10(d,1)
stats10(d,2)
stats10(d,3)
stats10(d,4)

abline(h = 0.09, col = "black", lty = 2, lwd = 0.1)
mtext("Together, 20 Btlncks -en large (SVM)", side = 3, line = -1.4, outer = TRUE, font = 2)
