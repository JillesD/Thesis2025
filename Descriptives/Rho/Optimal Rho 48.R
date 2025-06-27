# --------------------------------------
# Parallelized grid search + sparsity
# --------------------------------------

# 0. Install/load required packages
if (!require(glasso))    install.packages("glasso")
if (!require(zoo))       install.packages("zoo")
if (!require(lubridate)) install.packages("lubridate")
if (!require(doParallel)) install.packages("doParallel")
if (!require(foreach))   install.packages("foreach")

library(glasso)
library(zoo)
library(lubridate)
library(doParallel)
library(foreach)

# 1. Read & prepare the data (unchanged)
file_path <- "C:/Users/Jilles/OneDrive/Thesis/Data/Cleaned/48_Industry_Portfolios_excess_clean_rounded (1).csv"
data <- read.csv(file_path, stringsAsFactors = FALSE)

# Ensure there's a Date column
if (!("Date" %in% names(data))) {
  stop("No 'Date' column found; please adjust column name.")
}

# Parse Date; some entries might be "YYYY-MM" so we fall back on as.yearmon if needed
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

# 3. Build the ρ‐grid and preallocate storage
rho_values <- seq(0.1, 2.0, by = 0.1)
n_rho      <- length(rho_values)
avg_loglik <- numeric(n_rho)   # to store average log‐likelihood for each ρ

# 4. Pre‐construct a mask matrix (0 on diag, 1 off‐diag) for faster penalty‐matrix creation
p    <- ncol(X_all)
mask <- matrix(1, p, p)
diag(mask) <- 0  # now mask[i,i] = 0, mask[i,j≠i] = 1

# 5. Set up parallel backend (one worker for each ρ)
num_cores <- parallel::detectCores() - 1
cl <- makeCluster(num_cores)
registerDoParallel(cl)

# 6. Parallel loop over ρ values
avg_loglik <- foreach(r_idx = seq_len(n_rho), .combine = "c",
                      .packages = c("glasso")) %dopar% {
                        rho <- rho_values[r_idx]
                        loglik_vals <- numeric(n_eval)
                        
                        for (k in seq_len(n_eval)) {
                          j <- training_size + k
                          train_x <- X_all[(j - training_size):(j - 1), ]  # past 120 months
                          test_x  <- X_all[j, ]                            # single test observation
                          
                          # Compute sample covariance & mean on the training window
                          S  <- cov(train_x)
                          mu <- colMeans(train_x)
                          
                          # Build penalty matrix: zeros on diag, ρ on off‐diag
                          pmat <- rho * mask
                          
                          # Fit graphical lasso
                          gl    <- glasso(s = S, rho = pmat)
                          Omega <- gl$wi  # precision matrix estimate
                          
                          # Demean the test observation
                          r_tilde <- as.numeric(test_x - mu)
                          
                          # Compute log‐likelihood: log det(Omega) - r_tilde' Omega r_tilde
                          ld_val <- determinant(Omega, logarithm = TRUE)$modulus
                          quad  <- as.numeric(t(r_tilde) %*% Omega %*% r_tilde)
                          loglik_vals[k] <- ld_val - quad
                        }
                        
                        mean(loglik_vals)
                      }

# Stop the cluster once grid‐search is done
stopCluster(cl)

# 7. Identify the optimal ρ
best_idx     <- which.max(avg_loglik)
optimal_rho  <- rho_values[best_idx]

# 8. Print summary of ρ vs. average log‐likelihood, and show optimal value
df_summary <- data.frame(
  Rho              = rho_values,
  AvgLogLikelihood = round(avg_loglik, 6)
)

print(df_summary, row.names = FALSE)
cat(sprintf(
  "\nOptimal rho (off‐diag only): %.2f  (avg log‐likelihood = %.6f)\n",
  optimal_rho, avg_loglik[best_idx]
))

# 9. (Optional) Plot average log‐likelihood vs. ρ
plot(
  df_summary$Rho, df_summary$AvgLogLikelihood, type = "b",
  xlab = "Rho", ylab = "Average Log‐Likelihood",
  main = "Grid Search (Off‐Diag Penalty)"
)

# 10. Compute average sparsity (% zeros) at the chosen optimal ρ
zero_pct <- numeric(n_eval)

for (k in seq_len(n_eval)) {
  j       <- training_size + k
  train_x <- X_all[(j - training_size):(j - 1), ]
  
  S <- cov(train_x)
  pmat_opt <- optimal_rho * mask
  gl_opt   <- glasso(s = S, rho = pmat_opt)
  Omega_opt <- gl_opt$wi
  
  total_elements <- length(Omega_opt)
  zero_pct[k]    <- (sum(Omega_opt == 0) / total_elements) * 100
}

avg_sparsity <- mean(zero_pct)

cat(sprintf(
  "\nAverage sparsity (%% zeros per matrix) over %d estimates: %.2f%%\n",
  n_eval, avg_sparsity
))

