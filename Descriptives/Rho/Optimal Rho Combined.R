# Load necessary libraries
if(!require(glasso)) install.packages("glasso")
if(!require(zoo)) install.packages("zoo")
if(!require(lubridate)) install.packages("lubridate")
library(glasso)
library(zoo)
library(lubridate)

# Read and prepare data
file_path <- "C:/Users/Jilles/OneDrive/Thesis/Data/Cleaned/Combined_Portfolios_excess_clean_rounded.csv"
data <- read.csv(file_path, stringsAsFactors = FALSE)

# Parse Date column
if(!("Date" %in% names(data))) stop("No 'Date' column found; please adjust column name.")
data$Date <- as.Date(data$Date)
if(any(is.na(data$Date))) {
  data$Date <- as.Date(as.yearmon(data$Date, "%Y-%m"))
}
data <- data[order(data$Date), ]

# Extract numeric returns and dates
X_all <- data[, sapply(data, is.numeric)]
dates  <- data$Date

# Rolling-window parameters
training_size <- 120    # months for training
n_eval        <- 120    # number of evaluation months
start_eval    <- training_size + 1
end_eval      <- training_size + n_eval

# Data sufficiency check
if(nrow(X_all) < end_eval) {
  stop(sprintf("Data has %d months; need at least %d for rolling eval.", nrow(X_all), end_eval))
}

# Define grid of rho values\
rho_values <- seq(0.1, 2.0, by = 0.1)

# Prepare storage for average log-likelihoods
avg_loglik <- numeric(length(rho_values))

# Loop over rho grid
for(r_idx in seq_along(rho_values)) {
  rho <- rho_values[r_idx]
  loglik_vals <- numeric(n_eval)
  
  # Rolling estimation
  for(k in seq_len(n_eval)) {
    j <- training_size + k
    train_x <- X_all[(j - training_size):(j - 1), ]
    test_x  <- X_all[j, ]
    
    # sample covariance and mean
    S     <- cov(train_x)
    mu    <- colMeans(train_x)
    
    # build penalty matrix: rho on off-diagonals, 0 on diagonals
    pmat  <- matrix(rho, ncol(S), ncol(S))
    diag(pmat) <- 0
    
    # graphical lasso with off-diagonal penalty only
    gl    <- glasso(s = S, rho = pmat)
    Omega <- gl$wi
    
    # demeaned test
    r_tilde <- as.numeric(test_x - mu)
    
    # compute log-likelihood
    logdet <- as.numeric(determinant(Omega, logarithm = TRUE)$modulus)
    loglik_vals[k] <- logdet - as.numeric(t(r_tilde) %*% Omega %*% r_tilde)
  }
  
  # average over evaluation months
  avg_loglik[r_idx] <- mean(loglik_vals)
}

# Identify optimal rho
best_idx <- which.max(avg_loglik)
optimal_rho <- rho_values[best_idx]

# Prepare summary table
df_summary <- data.frame(
  Rho             = rho_values,
  AvgLogLikelihood = avg_loglik
)

# Output results
print(df_summary)
cat(sprintf("\nOptimal rho (off-diag only): %.2f (avg log-likelihood = %.4f)\n", 
            optimal_rho, avg_loglik[best_idx]))

# Optional: plot average log-likelihood vs. rho
plot(
  df_summary$Rho, df_summary$AvgLogLikelihood, type = "b",
  xlab = "Rho", ylab = "Average Log-Likelihood",
  main = "Grid Search (Off-Diag Penalty)"
)

# -----------------------------------------------------------
# Compute average sparsity (% zeros per matrix) at optimal rho
# -----------------------------------------------------------
zero_pct <- numeric(n_eval)

for(k in seq_len(n_eval)) {
  j <- training_size + k
  train_x <- X_all[(j - training_size):(j - 1), ]
  
  # sample covariance
  S <- cov(train_x)
  
  # penalty matrix at optimal rho
  pmat_opt <- matrix(optimal_rho, ncol(S), ncol(S))
  diag(pmat_opt) <- 0
  
  # graphical lasso with optimal rho
  gl_opt <- glasso(s = S, rho = pmat_opt)
  Omega_opt <- gl_opt$wi
  
  # percent zeros in this Omega
  total_elements <- length(Omega_opt)
  zero_pct[k] <- (sum(Omega_opt == 0) / total_elements) * 100
}

# Compute average sparsity percent
avg_sparsity <- mean(zero_pct)

# Report sparsity\ ncat(sprintf("\nAverage sparsity (%% zeros per matrix) over %d estimates: %.2f%%\n", n_eval, avg_sparsity))
