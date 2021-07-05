
install.packages('SLOPE')


DATASETS = c('ATV','APV','IDV','NFV','LPV','RTV','SQV')

FDRlevel = 0.2
library('SLOPE')


for (DATASET in DATASETS) {


load(paste0(paste0('../y',DATASET),'.RData'))
X <- read.csv(file = paste0(paste0('../X',DATASET),'.csv'))
X=X[,2:dim(X)[2]]

fit <- SLOPE(X, y, family = "gaussian", lambda = "gaussian",  solver = "admm", q = FDRlevel, alpha = "estimate")
selected_slope <- which(fit$nonzeros)



write.csv(selected_slope, file =paste0(paste0('slopeHIV-',DATASET),'.csv'))}