# cond_numbers_IND48.R
# ---------------------------------------
# Compute condition numbers (mean & sd) of inverse-covariance estimates
#  • Glasso (ρ = same as your script)
#  • Ledoit–Wolf constant-correlation shrinkage (via covCor)
# over the exact same rolling window you've used for IND48.

# 0) Packages
if (!require(glasso))   install.packages("glasso")
if (!require(zoo))      install.packages("zoo")
library(glasso)
library(zoo)

# --- Helper for rep.row ---
rep.row <- function(x, n) {
  matrix(rep(x, each = n), nrow = n)
}

# --- Covariance Estimator via Shrunk Correlation (Ledoit–Wolf) ---
covCor <- function(Y, k = -1) {
  dim.Y <- dim(Y); N <- dim.Y[1]; p <- dim.Y[2]
  if (k < 0) { Y <- scale(Y, scale = FALSE); k <- 1 }
  n      <- N - k
  sample <- (t(Y) %*% Y) / n
  
  # constant-correlation target
  samplevar <- diag(sample)
  sqrtvar   <- sqrt(samplevar)
  rBar      <- (sum(sample / outer(sqrtvar, sqrtvar)) - p) /
    (p * (p - 1))
  target    <- rBar * outer(sqrtvar, sqrtvar)
  diag(target) <- samplevar
  
  # π
  Y2       <- Y^2
  sample2  <- (t(Y2) %*% Y2) / n
  piMat    <- sample2 - sample^2
  pihat    <- sum(piMat)
  
  # γ
  gammahat <- norm(c(sample - target), type = "2")^2
  
  # ρ
  rho_diag <- sum(diag(piMat))
  term1    <- (t(Y^3) %*% Y) / n
  term2    <- rep.row(samplevar, p) * sample
  term2    <- t(term2)
  thetaMat <- term1 - term2
  diag(thetaMat) <- 0
  rho_off  <- rBar * sum(outer(1/sqrtvar, sqrtvar) * thetaMat)
  rhohat   <- rho_diag + rho_off
  
  # shrinkage intensity
  kappahat  <- (pihat - rhohat) / gammahat
  shrinkage <- max(0, min(1, kappahat / n))
  
  # estimator
  sigmahat <- shrinkage * target + (1 - shrinkage) * sample
  return(sigmahat)
}

# 1) Load your 48-industry returns
df <- read.csv(
  "C:/Users/Jilles/OneDrive/Thesis/Data/Cleaned/Combined_Portfolios_excess_clean_rounded.csv",
  stringsAsFactors = FALSE
)
df$Date <- as.Date(as.yearmon(df$Date, "%Y-%m"))
df <- df[order(df$Date), ]
X_all <- df[, sapply(df, is.numeric)]
dates <- df$Date

n_assets <- ncol(X_all)

# 2) Parameters (identical to your backtest)
rho_optimal <- 1.4
train_start <- as.Date(as.yearmon("1973-07"))
train_end   <- as.Date(as.yearmon("1983-06"))
last_test   <- as.Date(as.yearmon("2010-12"))

i_start <- which(dates == train_start)
i_end   <- which(dates == train_end)
if (length(i_start)==0 || length(i_end)==0)
  stop("Could not find training-window dates in your data")

training_size <- i_end - i_start + 1
test_idx_vec  <- which(dates > train_end & dates <= last_test)
n_win         <- length(test_idx_vec)

# 3) Preallocate storage
cond_glasso <- numeric(n_win)
cond_lw     <- numeric(n_win)

# 4) Rolling-window loop
window_end <- i_end
j <- 1
while (TRUE) {
  test_idx <- window_end + 1
  if (test_idx > length(dates) || dates[test_idx] > last_test) break
  
  # estimation window
  win_start <- window_end - training_size + 1
  Xw        <- X_all[win_start:window_end, ]
  
  # compute sample covariance once for Glasso
  S <- cov(Xw)
  
  # (a) Glasso
  gl <- glasso(s = S, rho = rho_optimal, penalize.diagonal = FALSE)
  Omega_g        <- gl$wi
  cond_glasso[j] <- kappa(Omega_g, exact = TRUE)
  
  # (b) Ledoit–Wolf constant-correlation shrinkage
  Sigma_lw   <- covCor(Xw)
  Omega_lw   <- solve(Sigma_lw)
  cond_lw[j] <- kappa(Omega_lw, exact = TRUE)
  
  # advance
  window_end <- window_end + 1
  j <- j + 1
}

# 5) Summarize
res <- data.frame(
  Method = c("Glasso (Ω̂)", "Ledoit–Wolf (Σ̂_LW⁻¹)"),
  Mean   = c(mean(cond_glasso), mean(cond_lw)),
  SD     = c(  sd(cond_glasso),   sd(cond_lw))
)

print(
  transform(res,
            Mean = round(Mean, 1),
            SD   = round(SD,   1))
)
