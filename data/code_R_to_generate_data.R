
# CHOOSE HERE THE NAME OF THE DATASET THAT YOU WANT TO SAVE
DATASETS = c('ATV','APV','IDV','NFV','LPV','RTV','SQV')


# The data preprocessing was taken from https://cran.r-project.org/web/packages/knockoff/vignettes/hiv.html


for (DATASET in DATASETS) {

drug_class = 'PI' # Possible drug types are 'PI', 'NRTI', and 'NNRTI'. 

base_url = 'http://hivdb.stanford.edu/pages/published_analysis/genophenoPNAS2006'
gene_url = paste(base_url, 'DATA', paste0(drug_class, '_DATA.txt'), sep='/')
tsm_url = paste(base_url, 'MUTATIONLISTS', 'NP_TSM', drug_class, sep='/')

gene_df = read.delim(gene_url, na.string = c('NA', ''), stringsAsFactors = FALSE)
tsm_df = read.delim(tsm_url, header = FALSE, stringsAsFactors = FALSE)
names(tsm_df) = c('Position', 'Mutations')

# Returns rows for which every column matches the given regular expression.
grepl_rows <- function(pattern, df) {
  cell_matches = apply(df, c(1,2), function(x) grepl(pattern, x))
  apply(cell_matches, 1, all)
}

pos_start = which(names(gene_df) == 'P1')
pos_cols = seq.int(pos_start, ncol(gene_df))
valid_rows = grepl_rows('^(\\.|-|[A-Zid]+)$', gene_df[,pos_cols])
gene_df = gene_df[valid_rows,]

# Flatten a matrix to a vector with names from concatenating row/column names.
flatten_matrix <- function(M, sep='.') {
  x <- c(M)
  names(x) <- c(outer(rownames(M), colnames(M),
                      function(...) paste(..., sep=sep)))
  x
}

# Construct preliminary design matrix.
muts = c(LETTERS, 'i', 'd')
X = outer(muts, as.matrix(gene_df[,pos_cols]), Vectorize(grepl))
X = aperm(X, c(2,3,1))
dimnames(X)[[3]] <- muts
X = t(apply(X, 1, flatten_matrix))
mode(X) <- 'numeric'

# Remove any mutation/position pairs that never appear in the data.
X = X[,colSums(X) != 0]

# Extract response matrix.
Y = gene_df[,4:(pos_start-1)]



y = Y[DATASET]

y = log(y)

# Remove patients with missing measurements.
missing = is.na(y)
y = y[!missing]
X = X[!missing,]

# Remove predictors that appear less than 3 times.
X = X[,colSums(X) >= 3]

# Remove duplicate predictors.
X = X[,colSums(abs(cor(X)-1) < 1e-4) == 1]


save(y,file=paste0(paste0('y',DATASET),'.RData'))
labels = colnames(X)
save(labels,file=paste0(paste0('labels',DATASET),'.RData'))
write.csv(X, paste0(paste0('X',DATASET),'.csv'))
truelabels = tsm_df$Position
save(truelabels,file='truelabels.RData')}


