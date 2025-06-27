# backtest_with_covCor_and_GELNET.R
# Rolling‐window GMV backtest including Glasso, Sample, JM, Equal, Cov-Cor shrinkage, and Graphical Elastic Net (GELNET, identity target)


# Load necessary libraries
if (!require(glasso))   install.packages("glasso")
if (!require(zoo))      install.packages("zoo")
if (!require(quadprog)) install.packages("quadprog")
if (!require(devtools)) install.packages("devtools")
if (!require(GLassoElnetFast)) devtools::install_github("TobiasRuckstuhl/GLassoElnetFast")
library(glasso); library(zoo); library(quadprog); library(GLassoElnetFast)

# --- Helper for rep.row ---
rep.row <- function(x, n) {
  matrix(rep(x, each = n), nrow = n)
}

# --- Covariance Estimator via Shrunk Correlation ---
covCor <- function(Y, k = -1) {
  dim.Y <- dim(Y); N <- dim.Y[1]; p <- dim.Y[2]
  if (k < 0) { Y <- scale(Y, scale = FALSE); k <- 1 }
  n      <- N - k
  sample <- (t(Y) %*% Y) / n
  samplevar <- diag(sample)
  sqrtvar   <- sqrt(samplevar)
  rBar      <- (sum(sample / outer(sqrtvar, sqrtvar)) - p) / (p * (p - 1))
  target    <- rBar * outer(sqrtvar, sqrtvar)
  diag(target) <- samplevar
  Y2       <- Y^2
  sample2  <- (t(Y2) %*% Y2) / n
  piMat    <- sample2 - sample^2
  pihat    <- sum(piMat)
  gammahat <- norm(c(sample - target), type = "2")^2
  rho_diag <- sum(diag(piMat))
  term1    <- (t(Y^3) %*% Y) / n
  term2    <- rep.row(samplevar, p) * sample
  term2    <- t(term2)
  thetaMat <- term1 - term2
  diag(thetaMat) <- 0
  rho_off  <- rBar * sum(outer(1/sqrtvar, sqrtvar) * thetaMat)
  rhohat   <- rho_diag + rho_off
  kappahat  <- (pihat - rhohat) / gammahat
  shrinkage <- max(0, min(1, kappahat / n))
  sigmahat <- shrinkage * target + (1 - shrinkage) * sample
  return(sigmahat)
}

# --- Read & prepare data ---
file_path <- "C:/Users/Jilles/OneDrive/Thesis/Data/Cleaned/48_Industry_Portfolios_excess_clean_rounded (1).csv"
data      <- read.csv(file_path, stringsAsFactors = FALSE)
data$Date <- as.Date(as.yearmon(data$Date, "%Y-%m"))
data      <- data[order(data$Date), ]
X_all     <- data[, sapply(data, is.numeric)]
dates     <- data$Date

# --- Parameters ---
rho_optimal  <- 1.3
train_start  <- as.Date(as.yearmon("1973-07")); train_end <- as.Date(as.yearmon("1983-06"))
last_test    <- as.Date(as.yearmon("2010-12"))
i_start      <- which(dates == train_start); i_end <- which(dates == train_end)
if (!length(i_start) || !length(i_end)) stop("Train window not found")

training_size <- i_end - i_start + 1
n_assets      <- ncol(X_all)
oneN          <- rep(1, n_assets)
ew_weights    <- rep(1/n_assets, n_assets)

# --- GELNET parameters (FIXED for now) ---
lambda_gelnet <- 0.9    # penalty parameter (try 0.05~0.2)
alpha_gelnet  <- 0.5      # elastic net mixing, 0 = Ridge, 1 = Lasso

# --- Storage ---
results <- data.frame(
  WindowEnd = as.Date(character()), TestDate = as.Date(character()),
  GMV = numeric(), SCovGMV = numeric(),
  GMVJM = numeric(), EW = numeric(), CCovGMV = numeric(), GELNET_GMV = numeric(),
  stringsAsFactors = FALSE
)
weights_gmv     <- weights_scov <- weights_gmv_jm <- weights_ccov_gmv <- weights_gelnet <- 
  matrix(NA, ncol = n_assets, nrow = 0)
colnames(weights_gmv) <- colnames(weights_scov) <- colnames(weights_gmv_jm) <-
  colnames(weights_ccov_gmv) <- colnames(weights_gelnet) <- colnames(X_all)

# --- Rolling window backtest ---
window_end <- i_end
while (TRUE) {
  test_idx <- window_end + 1
  if (test_idx > length(dates) || dates[test_idx] > last_test) break
  win_start <- window_end - training_size + 1
  data_win  <- X_all[win_start:window_end, ]
  
  # (1) Glasso GMV
  S      <- cov(data_win)
  gl     <- glasso(s = S, rho = rho_optimal, penalize.diagonal = FALSE)
  Omega  <- gl$wi
  w_gmv  <- as.numeric(Omega %*% oneN / as.numeric(t(oneN) %*% Omega %*% oneN))
  weights_gmv <- rbind(weights_gmv, w_gmv)
  
  # (2) Sample‐cov GMV
  Omega_sc <- solve(S)
  w_scov   <- as.numeric(Omega_sc %*% oneN / as.numeric(t(oneN) %*% Omega_sc %*% oneN))
  weights_scov <- rbind(weights_scov, w_scov)
  
  # (3) GMV‐JM (no‐short)
  Dmat <- 2 * S; dvec <- rep(0, n_assets)
  Amat <- cbind(oneN, diag(n_assets)); bvec <- c(1, rep(0, n_assets))
  sol   <- solve.QP(Dmat, dvec, Amat, bvec, meq = 1)$solution
  w_jm  <- sol / sum(sol)
  weights_gmv_jm <- rbind(weights_gmv_jm, w_jm)
  
  # (4) EW
  w_ew <- ew_weights
  
  # (5) Cov–Correlation GMV
  Sigma_cc <- cov(data_win)
  Omega_cc <- solve(Sigma_cc)
  w_ccov   <- as.numeric(Omega_cc %*% oneN / as.numeric(t(oneN) %*% Omega_cc %*% oneN))
  weights_ccov_gmv <- rbind(weights_ccov_gmv, w_ccov)
  
  # (6) Graphical Elastic Net (GELNET) - GMV (identity target)
  S_cor <- cor(data_win)
  gelnet_fit <- GLassoElnetFast::gelnet(
    S      = S_cor,
    lambda = lambda_gelnet,
    alpha  = alpha_gelnet,
    penalize.diagonal = FALSE,
  )
  
  Omega_gelnet <- gelnet_fit$Theta
  w_gelnet <- as.numeric(Omega_gelnet %*% oneN / as.numeric(t(oneN) %*% Omega_gelnet %*% oneN))
  weights_gelnet <- rbind(weights_gelnet, w_gelnet)
  
  # compute next‐month return
  r_next <- as.numeric(X_all[test_idx, ])
  ret_gmv    <- sum(w_gmv    * r_next)
  ret_scov   <- sum(w_scov   * r_next)
  ret_jm     <- sum(w_jm     * r_next)
  ret_ew     <- sum(w_ew     * r_next)
  ret_ccov   <- sum(w_ccov   * r_next)
  ret_gelnet <- sum(w_gelnet * r_next)
  
  results <- rbind(results, data.frame(
    WindowEnd = dates[window_end], TestDate = dates[test_idx],
    GMV = ret_gmv, SCovGMV = ret_scov,
    GMVJM = ret_jm, EW = ret_ew, CCovGMV = ret_ccov, GELNET_GMV = ret_gelnet
  ))
  
  window_end <- window_end + 1
}

# --- Performance evaluation & summary ---

# Performance function
evaluate_perf <- function(returns) {
  T        <- length(returns)
  Rbar_m   <- mean(returns)
  sigma2_m <- var(returns)
  sharpe_m <- Rbar_m / sqrt(sigma2_m)
  Rbar_a   <- Rbar_m * 12
  sigma2_a <- sigma2_m * 12
  sharpe_a <- Rbar_a / sqrt(sigma2_a)
  list(
    T = T,
    Rbar_m = Rbar_m, sigma2_m = sigma2_m, sharpe_m = sharpe_m,
    Rbar_a = Rbar_a, sigma2_a = sigma2_a, sharpe_a = sharpe_a
  )
}

# Quantile metrics & Herfindahl index
q_metrics <- function(wmat) {
  vec  <- as.vector(wmat)
  wq   <- quantile(vec, probs = c(0, .01, .05, .95, .99, 1))
  herf <- 10000 * mean(rowSums(wmat^2))
  list(wq = wq, herf = herf)
}

# turnover helper
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

# avg turnover → transaction cost
gamma <- 5
compute_TC <- function(TO_vec) {
  turn_m <- mean(TO_vec)
  turn_a <- turn_m * 12
  TC_a   <- turn_a * 0.005
  list(turn_m = turn_m, turn_a = turn_a, TC_a = TC_a)
}

# CER calculation
to_CER <- function(Rbar_a, sigma2_a, TC_a) {
  R_d  <- Rbar_a / 100
  v_d  <- sigma2_a / 10000
  CERd <- R_d - (gamma / 2) * v_d - TC_a
  CERd * 100
}

# compute perf & turnover for each strategy
eval_gmv     <- evaluate_perf(results$GMV)
eval_scov    <- evaluate_perf(results$SCovGMV)
eval_jm      <- evaluate_perf(results$GMVJM)
eval_ew      <- evaluate_perf(results$EW)
eval_ccov    <- evaluate_perf(results$CCovGMV)
eval_gelnet  <- evaluate_perf(results$GELNET_GMV)

weights_ew_mat <- matrix(rep(ew_weights, each = eval_ew$T),
                         nrow = eval_ew$T, byrow = FALSE)
metrics_gmv    <- q_metrics(weights_gmv)
metrics_scov   <- q_metrics(weights_scov)
metrics_jm     <- q_metrics(weights_gmv_jm)
metrics_ew     <- q_metrics(weights_ew_mat)
metrics_ccov   <- q_metrics(weights_ccov_gmv)
metrics_gelnet <- q_metrics(weights_gelnet)

test_start   <- i_end + 1
test_end     <- which(dates == last_test)
returns_oos  <- as.matrix(X_all[test_start:test_end, ])

TO_g   <- compute_TC(calc_TO(weights_gmv,       returns_oos))
TO_s   <- compute_TC(calc_TO(weights_scov,      returns_oos))
TO_j   <- compute_TC(calc_TO(weights_gmv_jm,    returns_oos))
TO_e   <- compute_TC(calc_TO(weights_ew_mat,    returns_oos))
TO_c   <- compute_TC(calc_TO(weights_ccov_gmv,  returns_oos))
TO_ge  <- compute_TC(calc_TO(weights_gelnet,    returns_oos))

CER_g  <- to_CER(eval_gmv$Rbar_a,     eval_gmv$sigma2_a,    TO_g$TC_a)
CER_s  <- to_CER(eval_scov$Rbar_a,    eval_scov$sigma2_a,   TO_s$TC_a)
CER_j  <- to_CER(eval_jm$Rbar_a,      eval_jm$sigma2_a,     TO_j$TC_a)
CER_e  <- to_CER(eval_ew$Rbar_a,      eval_ew$sigma2_a,     TO_e$TC_a)
CER_c  <- to_CER(eval_ccov$Rbar_a,    eval_ccov$sigma2_a,   TO_c$TC_a)
CER_ge <- to_CER(eval_gelnet$Rbar_a,  eval_gelnet$sigma2_a, TO_ge$TC_a)

# === Output summary ===
cat(sprintf("Out‐of‐sample periods: %d\n\n", eval_gmv$T))
# ... (same printing blocks as before, now with GELNET) ...

# 1) Glasso GMV
cat("1) Glasso GMV:\n")
cat(sprintf(" Monthly Ret: %.4f%%  Var: %.4f  Sharpe: %.4f\n",
            eval_gmv$Rbar_m, eval_gmv$sigma2_m, eval_gmv$sharpe_m))
cat(sprintf(" Annual  Ret: %.4f%%  Var: %.4f  Sharpe: %.4f\n",
            eval_gmv$Rbar_a, eval_gmv$sigma2_a, eval_gmv$sharpe_a))
cat(" WtQ (Min,1%,5%,95%,99%,Max): ",
    paste(round(metrics_gmv$wq, 4), collapse = ", "), "\n")
cat(sprintf(" Herf: %.2f  Turn M: %.4f  A: %.4f  CER: %.2f%%\n\n",
            metrics_gmv$herf, TO_g$turn_m, TO_g$turn_a, CER_g))

# 2) Sample‐Cov GMV
cat("2) Sample‐Cov GMV:\n")
cat(sprintf(" Monthly Ret: %.4f%%  Var: %.4f  Sharpe: %.4f\n",
            eval_scov$Rbar_m, eval_scov$sigma2_m, eval_scov$sharpe_m))
cat(sprintf(" Annual  Ret: %.4f%%  Var: %.4f  Sharpe: %.4f\n",
            eval_scov$Rbar_a, eval_scov$sigma2_a, eval_scov$sharpe_a))
cat(" WtQ (Min,1%,5%,95%,99%,Max): ",
    paste(round(metrics_scov$wq, 4), collapse = ", "), "\n")
cat(sprintf(" Herf: %.2f  Turn M: %.4f  A: %.4f  CER: %.2f%%\n\n",
            metrics_scov$herf, TO_s$turn_m, TO_s$turn_a, CER_s))

# 3) GMV‐JM (no‐short)
cat("3) GMV‐JM (Jagannathan & Ma, 2003):\n")
cat(sprintf(" Monthly Ret: %.4f%%  Var: %.4f  Sharpe: %.4f\n",
            eval_jm$Rbar_m, eval_jm$sigma2_m, eval_jm$sharpe_m))
cat(sprintf(" Annual  Ret: %.4f%%  Var: %.4f  Sharpe: %.4f\n",
            eval_jm$Rbar_a, eval_jm$sigma2_a, eval_jm$sharpe_a))
cat(" WtQ (Min,1%,5%,95%,99%,Max): ",
    paste(round(metrics_jm$wq, 4), collapse = ", "), "\n")
cat(sprintf(" Herf: %.2f  Turn M: %.4f  A: %.4f  CER: %.2f%%\n\n",
            metrics_jm$herf, TO_j$turn_m, TO_j$turn_a, CER_j))

# 4) Equal‐Weight
cat("4) Equal‐Weight:\n")
cat(sprintf(" Monthly Ret: %.4f%%  Var: %.4f  Sharpe: %.4f\n",
            eval_ew$Rbar_m, eval_ew$sigma2_m, eval_ew$sharpe_m))
cat(sprintf(" Annual  Ret: %.4f%%  Var: %.4f  Sharpe: %.4f\n",
            eval_ew$Rbar_a, eval_ew$sigma2_a, eval_ew$sharpe_a))
cat(" WtQ (Min,1%,5%,95%,99%,Max): ",
    paste(round(metrics_ew$wq, 4), collapse = ", "), "\n")
cat(sprintf(" Herf: %.2f  Turn M: %.4f  A: %.4f  CER: %.2f%%\n\n",
            metrics_ew$herf, TO_e$turn_m, TO_e$turn_a, CER_e))

# 5) Cov‐Correlation GMV
cat("5) Ledoit & Wolf - GMV:\n")
cat(sprintf(" Monthly Ret: %.4f%%  Var: %.4f  Sharpe: %.4f\n",
            eval_ccov$Rbar_m, eval_ccov$sigma2_m, eval_ccov$sharpe_m))
cat(sprintf(" Annual  Ret: %.4f%%  Var: %.4f  Sharpe: %.4f\n",
            eval_ccov$Rbar_a, eval_ccov$sigma2_a, eval_ccov$sharpe_a))
cat(" WtQ (Min,1%,5%,95%,99%,Max): ",
    paste(round(metrics_ccov$wq, 4), collapse = ", "), "\n")


# 6) Graphical Elastic Net (identity target)
cat("6) Graphical Elastic Net (identity target):\n")
cat(sprintf(" Monthly Ret: %.4f%%  Var: %.4f  Sharpe: %.4f\n",
            eval_gelnet$Rbar_m, eval_gelnet$sigma2_m, eval_gelnet$sharpe_m))
cat(sprintf(" Annual  Ret: %.4f%%  Var: %.4f  Sharpe: %.4f\n",
            eval_gelnet$Rbar_a, eval_gelnet$sigma2_a, eval_gelnet$sharpe_a))
cat(" WtQ (Min,1%,5%,95%,99%,Max): ",
    paste(round(metrics_gelnet$wq, 4), collapse = ", "), "\n")
cat(sprintf(" Herf: %.2f  Turn M: %.4f  A: %.4f  CER: %.2f%%\n",
            metrics_gelnet$herf, TO_ge$turn_m, TO_ge$turn_a, CER_ge))

