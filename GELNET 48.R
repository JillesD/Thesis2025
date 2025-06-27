# backtest_with_2_GELNET_variants.R
# Rolling‐window GMV backtest using two variants of Graphical Elastic Net (GELNET)
# via the GLassoElnetFast package (a.k.a. GLassoGELNETfast):
#   1) GELNET on covariance, no target
#   2) GELNET on correlation, no target

# -----------------------------
# Load necessary libraries
# -----------------------------
if (!require(GLassoElnetFast)) {
  if (!require(devtools)) install.packages("devtools")
  devtools::install_github("TobiasRuckstuhl/GLassoElnetFast")
}
if (!require(zoo))      install.packages("zoo")
if (!require(quadprog)) install.packages("quadprog")

library(GLassoElnetFast)  # Provides gelnet()
library(zoo)
library(quadprog)

# -----------------------------
# Helper: replicate a row vector
# -----------------------------
rep.row <- function(x, n) {
  matrix(rep(x, each = n), nrow = n)
}

# -----------------------------
# Read & prepare data
# -----------------------------
file_path <- "C:/Users/Jilles/OneDrive/Thesis/Data/Cleaned/48_Industry_Portfolios_excess_clean_rounded (1).csv"
data      <- read.csv(file_path, stringsAsFactors = FALSE)

# Convert Date from "YYYY-MM" to Date (first day of month)
data$Date <- as.Date(as.yearmon(data$Date, "%Y-%m"))
data      <- data[order(data$Date), ]

# Extract numeric returns and dates
X_all <- data[, sapply(data, is.numeric)]
dates <- data$Date

# -----------------------------
# Parameters
# -----------------------------
lambda_gelnetCov <- 2.6
lambda_gelnetCor <- 0.9    # GELNET penalty parameter
alpha_gelnet  <- 0.5    # mixing: 0 = Ridge, 1 = Lasso

# Rolling‐window setup
train_start <- as.Date(as.yearmon("1973-07"))
train_end   <- as.Date(as.yearmon("1983-06"))
last_test   <- as.Date(as.yearmon("2010-12"))

i_start <- which(dates == train_start)
i_end   <- which(dates == train_end)
if (!length(i_start) || !length(i_end)) {
  stop("Training window not found. Check train_start / train_end.")
}

training_size <- i_end - i_start + 1
n_assets      <- ncol(X_all)
oneN          <- rep(1, n_assets)

# -----------------------------
# Storage for results & weights
# -----------------------------
results <- data.frame(
  WindowEnd             = as.Date(character()),
  TestDate              = as.Date(character()),
  GELNET_Cov_NoTarget   = numeric(),
  GELNET_Cor_NoTarget   = numeric(),
  stringsAsFactors = FALSE
)

weights_cov_no_tgt <- matrix(NA, ncol = n_assets, nrow = 0)
weights_cor_no_tgt <- matrix(NA, ncol = n_assets, nrow = 0)

colnames(weights_cov_no_tgt) <- colnames(X_all)
colnames(weights_cor_no_tgt) <- colnames(X_all)

# -----------------------------
# Rolling‐window backtest
# -----------------------------
window_end <- i_end

while (TRUE) {
  test_idx <- window_end + 1
  if (test_idx > length(dates) || dates[test_idx] > last_test) {
    break
  }
  win_start <- window_end - training_size + 1
  data_win  <- X_all[win_start:window_end, ]
  
  # --- 1) GELNET on covariance, no target ---
  S_cov      <- cov(data_win)
  gel_cov_nt <- GLassoElnetFast::gelnet(
    S                 = S_cov,
    lambda            = lambda_gelnetCov,
    alpha             = alpha_gelnet,
    penalize.diagonal = FALSE
  )
  Omega_cov_nt <- gel_cov_nt$Theta
  w_cov_nt     <- as.numeric(
    Omega_cov_nt %*% oneN / as.numeric(t(oneN) %*% Omega_cov_nt %*% oneN)
  )
  weights_cov_no_tgt <- rbind(weights_cov_no_tgt, w_cov_nt)
  
  # --- 2) GELNET on correlation, no target ---
  S_cor      <- cor(data_win)
  gel_cor_nt <- GLassoElnetFast::gelnet(
    S                 = S_cor,
    lambda            = lambda_gelnetCor,
    alpha             = alpha_gelnet,
    penalize.diagonal = FALSE
  )
  Omega_cor_nt <- gel_cor_nt$Theta
  w_cor_nt     <- as.numeric(
    Omega_cor_nt %*% oneN / as.numeric(t(oneN) %*% Omega_cor_nt %*% oneN)
  )
  weights_cor_no_tgt <- rbind(weights_cor_no_tgt, w_cor_nt)
  
  # -----------------------------
  # Compute next‐month returns
  # -----------------------------
  r_next <- as.numeric(X_all[test_idx, ])
  ret_cov_nt <- sum(w_cov_nt * r_next)
  ret_cor_nt <- sum(w_cor_nt * r_next)
  
  results <- rbind(
    results,
    data.frame(
      WindowEnd           = dates[window_end],
      TestDate            = dates[test_idx],
      GELNET_Cov_NoTarget = ret_cov_nt,
      GELNET_Cor_NoTarget = ret_cor_nt,
      stringsAsFactors = FALSE
    )
  )
  
  window_end <- window_end + 1
}

# -----------------------------
# Performance evaluation functions
# -----------------------------
evaluate_perf <- function(returns) {
  T        <- length(returns)
  Rbar_m   <- mean(returns)
  sigma2_m <- var(returns)
  sharpe_m <- if (sigma2_m > 0) Rbar_m / sqrt(sigma2_m) else NA
  Rbar_a   <- Rbar_m * 12
  sigma2_a <- sigma2_m * 12
  sharpe_a <- if (sigma2_a > 0) Rbar_a / sqrt(sigma2_a) else NA
  list(
    T        = T,
    Rbar_m   = Rbar_m,
    sigma2_m = sigma2_m,
    sharpe_m = sharpe_m,
    Rbar_a   = Rbar_a,
    sigma2_a = sigma2_a,
    sharpe_a = sharpe_a
  )
}

q_metrics <- function(wmat) {
  vec  <- as.vector(wmat)
  wq   <- quantile(vec, probs = c(0, .01, .05, .95, .99, 1))
  herf <- 10000 * mean(rowSums(wmat^2))
  list(wq = wq, herf = herf)
}

calc_TO <- function(w_mat, returns_oos) {
  M  <- nrow(w_mat)
  TO <- numeric(M - 1)
  for (t in 1:(M - 1)) {
    w_t      <- w_mat[t, ]
    r_t      <- returns_oos[t, ] / 100
    w_t_plus <- w_t * (1 + r_t) / sum(w_t * (1 + r_t))
    TO[t]    <- sum(abs(w_mat[t + 1, ] - w_t_plus))
  }
  TO
}

compute_TC <- function(TO_vec) {
  turn_m <- mean(TO_vec)
  turn_a <- turn_m * 12
  TC_a   <- turn_a * 0.005
  list(turn_m = turn_m, turn_a = turn_a, TC_a = TC_a)
}

to_CER <- function(Rbar_a, sigma2_a, TC_a, gamma = 5) {
  R_d  <- Rbar_a / 100
  v_d  <- sigma2_a / 10000
  CERd <- R_d - (gamma / 2) * v_d - TC_a
  CERd * 100
}

# -----------------------------
# Evaluate performance & turnover for each GELNET variant
# -----------------------------
ret_cov_nt_vec <- results$GELNET_Cov_NoTarget
ret_cor_nt_vec <- results$GELNET_Cor_NoTarget

perf_cov_nt <- evaluate_perf(ret_cov_nt_vec)
perf_cor_nt <- evaluate_perf(ret_cor_nt_vec)

# Subset returns_oos matrix for turnover computations
test_start   <- i_end + 1
test_end     <- which(dates == last_test)
returns_oos  <- as.matrix(X_all[test_start:test_end, ])

TO_cov_nt <- compute_TC(calc_TO(weights_cov_no_tgt, returns_oos))
TO_cor_nt <- compute_TC(calc_TO(weights_cor_no_tgt, returns_oos))

CER_cov_nt <- to_CER(perf_cov_nt$Rbar_a, perf_cov_nt$sigma2_a, TO_cov_nt$TC_a)
CER_cor_nt <- to_CER(perf_cor_nt$Rbar_a, perf_cor_nt$sigma2_a, TO_cor_nt$TC_a)

# -----------------------------
# Output summary
# -----------------------------
cat(sprintf("Out‐of‐sample periods (months): %d\n\n", perf_cov_nt$T))

# 1) GELNET (Covariance, no target)
cat("1) GELNET (Covariance, no target):\n")
cat(sprintf("   Monthly Ret: %.4f%%   Var: %.4f   Sharpe: %.4f\n",
            perf_cov_nt$Rbar_m, perf_cov_nt$sigma2_m, perf_cov_nt$sharpe_m))
cat(sprintf("   Annual  Ret: %.4f%%   Var: %.4f   Sharpe: %.4f\n",
            perf_cov_nt$Rbar_a, perf_cov_nt$sigma2_a, perf_cov_nt$sharpe_a))
qm_cov <- q_metrics(weights_cov_no_tgt)
cat("   WtQ (Min,1%,5%,95%,99%,Max): ",
    paste(round(qm_cov$wq, 4), collapse = ", "), "\n")
cat(sprintf("   Herf: %.2f   Turn M: %.4f   A: %.4f   CER: %.2f%%\n\n",
            qm_cov$herf, TO_cov_nt$turn_m, TO_cov_nt$turn_a, CER_cov_nt))

# 2) GELNET (Correlation, no target)
cat("2) GELNET (Correlation, no target):\n")
cat(sprintf("   Monthly Ret: %.4f%%   Var: %.4f   Sharpe: %.4f\n",
            perf_cor_nt$Rbar_m, perf_cor_nt$sigma2_m, perf_cor_nt$sharpe_m))
cat(sprintf("   Annual  Ret: %.4f%%   Var: %.4f   Sharpe: %.4f\n",
            perf_cor_nt$Rbar_a, perf_cor_nt$sigma2_a, perf_cor_nt$sharpe_a))
qm_cor <- q_metrics(weights_cor_no_tgt)
cat("   WtQ (Min,1%,5%,95%,99%,Max): ",
    paste(round(qm_cor$wq, 4), collapse = ", "), "\n")
cat(sprintf("   Herf: %.2f   Turn M: %.4f   A: %.4f   CER: %.2f%%\n",
            qm_cor$herf, TO_cor_nt$turn_m, TO_cor_nt$turn_a, CER_cor_nt))

