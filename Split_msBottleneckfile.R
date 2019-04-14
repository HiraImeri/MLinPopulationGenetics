#### INPUT PARAMETERS
input.file <- "tog20btlnkcsgnmcrngthta.msout"
dat <- readLines(input.file)

info <- strsplit(dat[1], split=" ")[[1]]
rep.lines <- grep("//", dat)

# take info directly from first line of output file -- # ONLY WORKS WHEN ALL no additional parameters in first part of command
num.reps <- as.numeric(info[3])
num.pops <- as.numeric(info[which(info == "-I")+1])
samp.sizes <- c(as.numeric(info[(which(info == "-I")+2):(which(info == "-I")+(1+num.pops))]))
theta <- as.numeric(info[which(info == "-t")+1])
rho <- as.numeric(info[which(info == "-r")+1])
seq.length <- as.numeric(info[which(info == "-r")+2])

# otherwise, manually override with known params
# num.reps <- 
# num.pops <- 
# samp.sizes <- 
####


# to put at the top of each split file
first.lines <- dat[1:2]
seg.sites <- dat[rep.lines+1]
positions <- dat[rep.lines+2]

positions.split = strsplit(positions, " ")
positions.split = lapply(positions.split, "[", -1) # get rid of "positions:" label at front
#positions.split = lapply(positions.split, as.numeric) # only if I want numbers



# split per population
for(pop.id in seq_len(num.pops)) {
  
  # go through each replicate for that given pop
  all.reps <-  sapply(seq_len(num.reps), function(rep.id) {
    hap.start <- rep.lines[rep.id] + 3 + (samp.sizes[pop.id] * (pop.id-1)) # first starting haplotype line
    
    pop.sample.haplotypes <- dat[hap.start:(hap.start+samp.sizes[pop.id]-1)]
    
    # remove monomorphic sites
    pop.sample.haplotypes <- do.call("rbind", strsplit(pop.sample.haplotypes, ""))
    counts <- sapply(apply(pop.sample.haplotypes, MARGIN=2, FUN=unique), FUN=length)
    monomorphic <- counts == 1
    poly.sample.haplotypes <- pop.sample.haplotypes[, !monomorphic]
    temp.seg.sites <- dim(poly.sample.haplotypes)[2]
    temp.positions <- positions.split[[rep.id]][!monomorphic]
    
    return(paste(c(
      sprintf("segsites: %i", temp.seg.sites),
      sprintf("positions: %s", paste(temp.positions, collapse= " ")),
      apply(poly.sample.haplotypes, MARGIN=1, FUN=paste, collapse = "")
    ), collapse= "\n"))
  }) 
  
  new.first.lines <- paste(c("./ms", samp.sizes[pop.id], num.reps, "-t", theta, "-r", rho, seq.length, "-I", "1", samp.sizes[pop.id], "0 remaining-text-here-should-be-irrelevant?"), collapse=" ")
  
  all.reps = c(paste(new.first.lines, collapse="\n"), all.reps)
  all.reps = paste(all.reps, collapse = "\n\n//\n")
  
  # write msOut file for this population
  cat(all.reps, file=sprintf("tog20btlnkcsMiggnmcrngthta_%i.msout", pop.id))
}


################################################################
#### INPUT PARAMETERS
input.file <- "originalmodeltrain.msout"
dat <- readLines(input.file)

info <- strsplit(dat[1], split=" ")[[1]]
rep.lines <- grep("//", dat)

# take info directly from first line of output file -- # ONLY WORKS WHEN ALL no additional parameters in first part of command
num.reps <- as.numeric(info[3])
num.pops <- as.numeric(info[which(info == "-I")+1])
samp.sizes <- c(as.numeric(info[(which(info == "-I")+2):(which(info == "-I")+(1+num.pops))]))
theta <- as.numeric(info[which(info == "-t")+1])
rho <- as.numeric(info[which(info == "-r")+1])
seq.length <- as.numeric(info[which(info == "-r")+2])

# otherwise, manually override with known params
# num.reps <- 
# num.pops <- 
# samp.sizes <- 
####


# to put at the top of each split file
first.lines <- dat[1:2]
seg.sites <- dat[rep.lines+1]
positions <- dat[rep.lines+2]

positions.split = strsplit(positions, " ")
positions.split = lapply(positions.split, "[", -1) # get rid of "positions:" label at front
#positions.split = lapply(positions.split, as.numeric) # only if I want numbers



# split per population
for(pop.id in seq_len(num.pops)) {
  
  # go through each replicate for that given pop
  all.reps <-  sapply(seq_len(num.reps), function(rep.id) {
    hap.start <- rep.lines[rep.id] + 3 + (samp.sizes[pop.id] * (pop.id-1)) # first starting haplotype line
    
    pop.sample.haplotypes <- dat[hap.start:(hap.start+samp.sizes[pop.id]-1)]
    
    # remove monomorphic sites
    pop.sample.haplotypes <- do.call("rbind", strsplit(pop.sample.haplotypes, ""))
    counts <- sapply(apply(pop.sample.haplotypes, MARGIN=2, FUN=unique), FUN=length)
    monomorphic <- counts == 1
    poly.sample.haplotypes <- pop.sample.haplotypes[, !monomorphic]
    temp.seg.sites <- dim(poly.sample.haplotypes)[2]
    temp.positions <- positions.split[[rep.id]][!monomorphic]
    
    return(paste(c(
      sprintf("segsites: %i", temp.seg.sites),
      sprintf("positions: %s", paste(temp.positions, collapse= " ")),
      apply(poly.sample.haplotypes, MARGIN=1, FUN=paste, collapse = "")
    ), collapse= "\n"))
  }) 
  
  new.first.lines <- paste(c("./ms", samp.sizes[pop.id], num.reps, "-t", theta, "-r", rho, seq.length, "-I", "1", samp.sizes[pop.id], "0 remaining-text-here-should-be-irrelevant?"), collapse=" ")
  
  all.reps = c(paste(new.first.lines, collapse="\n"), all.reps)
  all.reps = paste(all.reps, collapse = "\n\n//\n")
  
  # write msOut file for this population
  cat(all.reps, file=sprintf("originalmodeltrain_%i.msout", pop.id))
}

