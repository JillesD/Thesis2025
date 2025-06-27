# --------------------------------------
# Grid search for optimal λ for GELNET (correlation and covariance)
# with reliable progress indicators
# --------------------------------------

# 0. Install/load required packages
if (!require(GLassoElnetFast)) {
  if (!require(devtools)) install.packages("devtools")
  devtools::install_github("TobiasRuckstuhl/GLassoElnetFast")
}
if (!require(zoo))        install.packages("zoo")
if (!require(lubridate))  install.packages("lubridate")
if (!require(doSNOW))     install.packages("doSNOW")
if (!require(foreach))    install.packages("foreach")

library(GLassoElnetFast)
library(zoo)
library(lubridate)
library(doSNOW)
library(foreach)

# 1. Read & prepare the data (unchanged)
file_path <- "C:/Users/Jilles/OneDrive/Thesis/Data/Cleaned/100_Portfolios_10x10_excess_clean_rounded.csv"
data <- read.csv(file_path, stringsAsFactors = FALSE)

# Ensure there's a Date column
if (!("Date" %in% names(data))) {
  stop("No 'Date' column found; please adjust column name.")
}

# Parse Date; some entries might be "YYYY-MM" so fall back on as.yearmon if needed
data$Date <- as.Date(data$Date)
if (any(is.na(data$Date))) {
  data$Date <- as.Date(as.yearmon(data$Date, "%Y-%m"))
}

# Sort by date
data <- data[order(data$Date), ]

# Extract numeric returns (X_all) and dates vector
X_all <- data[, sapply(data, is.numeric)]
dates <- data$Date

# 2. Set rolling‐window parameters
training_size <- 120   # months used for each training window
n_eval        <- 120   # number of rolling‐window test points

# Ensure enough data
if (nrow(X_all) < (training_size + n_eval)) {
  stop(sprintf(
    "Data has %d rows; need at least %d (training_size + n_eval).",
    nrow(X_all), training_size + n_eval
  ))
}

# 3. Build the λ‐grid and preallocate storage
lambda_values <- seq(0.1, 2.0, by = 0.1)
n_lambda      <- length(lambda_values)

# Containers for average log‐likelihood for each λ
avg_loglik_cor <- numeric(n_lambda)
avg_loglik_cov <- numeric(n_lambda)

# Fix alpha at 0.5 for GELNET
alpha_gelnet <- 0.5

# 4. Set up parallel backend for correlation‐based GELNET
num_cores <- parallel::detectCores() - 1
cl <- makeCluster(num_cores, type = "SOCK")
registerDoSNOW(cl)

# 5a. Parallel loop over λ for correlation‐based GELNET (with progress bar)
cat("Grid search: correlation‐based GELNET\n")
pb_cor <- txtProgressBar(min = 0, max = n_lambda, style = 3)
progress_cor <- function(n) setTxtProgressBar(pb_cor, n)
opts_cor <- list(progress = progress_cor)

avg_loglik_cor <- foreach(l_idx = seq_len(n_lambda),
                          .combine = "c",
                          .packages = c("GLassoElnetFast", "zoo"),
                          .options.snow = opts_cor) %dopar% {
                            lambda_val  <- lambda_values[l_idx]
                            loglik_vals <- numeric(n_eval)
                            
                            for (k in seq_len(n_eval)) {
                              j       <- training_size + k
                              train_x <- X_all[(j - training_size):(j - 1), ]
                              test_x  <- X_all[j, ]
                              
                              S_cor <- cor(train_x)
                              mu    <- colMeans(train_x)
                              
                              gelnet_fit <- GLassoElnetFast::gelnet(
                                S                 = S_cor,
                                lambda            = lambda_val,
                                alpha             = alpha_gelnet,
                                penalize.diagonal = FALSE
                              )
                              Omega <- gelnet_fit$Theta
                              
                              r_tilde <- as.numeric(test_x - mu)
                              ld_val  <- as.numeric(determinant(Omega, logarithm = TRUE)$modulus)
                              quad    <- as.numeric(t(r_tilde) %*% Omega %*% r_tilde)
                              loglik_vals[k] <- ld_val - quad
                            }
                            
                            mean(loglik_vals)
                          }

close(pb_cor)
stopCluster(cl)

# 5b. Sequential loop over λ for covariance‐based GELNET (with progress bar)
cat("\nGrid search: covariance‐based GELNET\n")
pb_cov <- txtProgressBar(min = 0, max = n_lambda, style = 3)
for (l_idx in seq_len(n_lambda)) {
  lambda_val  <- lambda_values[l_idx]
  loglik_vals <- numeric(n_eval)
  
  for (k in seq_len(n_eval)) {
    j       <- training_size + k
    train_x <- X_all[(j - training_size):(j - 1), ]
    test_x  <- X_all[j, ]
    
    S_cov <- cov(train_x)
    mu    <- colMeans(train_x)
    
    gelnet_fit <- GLassoElnetFast::gelnet(
      S                 = S_cov,
      lambda            = lambda_val,
      alpha             = alpha_gelnet,
      penalize.diagonal = FALSE
    )
    Omega <- gelnet_fit$Theta
    
    r_tilde <- as.numeric(test_x - mu)
    ld_val  <- as.numeric(determinant(Omega, logarithm = TRUE)$modulus)
    quad    <- as.numeric(t(r_tilde) %*% Omega %*% r_tilde)
    loglik_vals[k] <- ld_val - quad
  }
  
  avg_loglik_cov[l_idx] <- mean(loglik_vals)
  setTxtProgressBar(pb_cov, l_idx)
}
close(pb_cov)

# 6. Identify the optimal λ for each method
best_idx_cor      <- which.max(avg_loglik_cor)
optimal_lambda_cor <- lambda_values[best_idx_cor]

best_idx_cov      <- which.max(avg_loglik_cov)
optimal_lambda_cov <- lambda_values[best_idx_cov]

# 7. Print summary of λ vs. average log‐likelihood for both methods
df_summary <- data.frame(
  Lambda                  = lambda_values,
  AvgLogLik_Correlation   = round(avg_loglik_cor, 6),
  AvgLogLik_Covariance    = round(avg_loglik_cov, 6)
)

print(df_summary, row.names = FALSE)
cat(sprintf(
  "\nOptimal λ (α = %.1f) for correlation‐based GELNET: %.2f  (avg log‐likelihood = %.6f)\n",
  alpha_gelnet, optimal_lambda_cor, avg_loglik_cor[best_idx_cor]
))
cat(sprintf(
  "Optimal λ (α = %.1f) for covariance‐based GELNET:  %.2f  (avg log‐likelihood = %.6f)\n",
  alpha_gelnet, optimal_lambda_cov, avg_loglik_cov[best_idx_cov]
))

# 8. (Optional) Plot average log‐likelihood vs. λ for both methods
plot(
  df_summary$Lambda, df_summary$AvgLogLik_Correlation, type = "b", col = "blue",
  xlab = "Lambda", ylab = "Average Log‐Likelihood",
  ylim = range(df_summary[, 2:3]),
  main = "Grid Search (Elastic‐Net Penalty, α = 0.5)"
)
lines(df_summary$Lambda, df_summary$AvgLogLik_Covariance, type = "b", col = "red")
legend(
  "bottomright",
  legend = c("Correlation‐based", "Covariance‐based"),
  col    = c("blue", "red"),
  lty    = 1, pch = 1
)

# 9. Compute average sparsity (% zeros) at the chosen optimal λ for both methods
zero_pct_cor <- numeric(n_eval)
zero_pct_cov <- numeric(n_eval)

for (k in seq_len(n_eval)) {
  j       <- training_size + k
  train_x <- X_all[(j - training_size):(j - 1), ]
  
  # (a) Correlation‐based at optimal λ
  S_cor <- cor(train_x)
  gelnet_opt_cor <- GLassoElnetFast::gelnet(
    S                 = S_cor,
    lambda            = optimal_lambda_cor,
    alpha             = alpha_gelnet,
    penalize.diagonal = FALSE
  )
  Omega_opt_cor      <- gelnet_opt_cor$Theta
  total_elements_cor <- length(Omega_opt_cor)
  zero_pct_cor[k]    <- (sum(Omega_opt_cor == 0) / total_elements_cor) * 100
  
  # (b) Covariance‐based at optimal λ
  S_cov <- cov(train_x)
  gelnet_opt_cov <- GLassoElnetFast::gelnet(
    S                 = S_cov,
    lambda            = optimal_lambda_cov,
    alpha             = alpha_gelnet,
    penalize.diagonal = FALSE
  )
  Omega_opt_cov      <- gelnet_opt_cov$Theta
  total_elements_cov <- length(Omega_opt_cov)
  zero_pct_cov[k]    <- (sum(Omega_opt_cov == 0) / total_elements_cov) * 100
}

avg_sparsity_cor <- mean(zero_pct_cor)
avg_sparsity_cov <- mean(zero_pct_cov)

cat(sprintf(
  "\nAverage sparsity (%% zeros per precision matrix) over %d estimates at λ = %.2f (correlation‐based): %.2f%%\n",
  n_eval, optimal_lambda_cor, avg_sparsity_cor
))
cat(sprintf(
  "Average sparsity (%% zeros per precision matrix) over %d estimates at λ = %.2f (covariance‐based):  %.2f%%\n",
  n_eval, optimal_lambda_cov, avg_sparsity_cov
))


