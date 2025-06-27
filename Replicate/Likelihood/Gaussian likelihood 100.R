# loglik_diff_IND48.R
# ---------------------------------------
# Compute avg log-likelihood differences for
#  • Glasso (fixed rho)
#  • Sample-cov precision
#  • Ledoit–Wolf constant-correlation precision
# on the IND48 rolling-window,
# using a single R̄ computed over 1983-07:2010-12.

# 0) Packages
if(!require(glasso))   install.packages("glasso")
if(!require(zoo))      install.packages("zoo")
library(glasso); library(zoo)

# 1) Helper: Ledoit–Wolf constant-correlation shrinkage
rep.row <- function(x,n) matrix(rep(x,each=n),nrow=n)
covCor <- function(Y,k=-1){
  D <- dim(Y); N<-D[1]; p<-D[2]
  if(k<0){ Y<-scale(Y,scale=FALSE); k<-1 }
  n<-N-k; S<-(t(Y)%*%Y)/n
  sv<-diag(S); sq<-sqrt(sv)
  rBar<-(sum(S/outer(sq,sq))-p)/(p*(p-1))
  Tgt<-rBar*outer(sq,sq); diag(Tgt)<-sv
  Y2<-(Y^2); S2<-(t(Y2)%*%Y2)/n; piM<-S2-S^2; pihat<-sum(piM)
  gamma<-norm(c(S-Tgt),"2")^2
  rho_d<-sum(diag(piM))
  t1<-(t(Y^3)%*%Y)/n
  t2<-t(rep.row(sv,p)*S); theta<-t1-t2; diag(theta)<-0
  rho_off<-rBar*sum(outer(1/sq,sq)*theta)
  kapp<-(pihat - (rho_d+rho_off))/gamma; κ<-max(0,min(1,kapp/n))
  κ*Tgt + (1-κ)*S
}

# 2) Load data
df <- read.csv(
  "C:/Users/Jilles/OneDrive/Thesis/Data/Cleaned/100_Portfolios_10x10_excess_clean_rounded.csv",
  stringsAsFactors=FALSE
)
df$Date <- as.Date(df$Date)
if(any(is.na(df$Date))) df$Date <- as.Date(as.yearmon(df$Date,"%Y-%m"))
df <- df[order(df$Date), ]
X_all <- df[, sapply(df, is.numeric)]
dates <- df$Date

# 3) Parameters & indices
rho_optimal <- 1.1             # your chosen rho
train_start <- as.Date("1973-07-01")
train_end   <- as.Date("1983-06-01")
last_test   <- as.Date("2010-12-01")

i_start <- which(dates==train_start)
i_end   <- which(dates==train_end)
if(!length(i_start)||!length(i_end)) stop("Train window not found")

# out-of-sample indices
test_idxs <- which(dates > train_end & dates <= last_test)
n_oos     <- length(test_idxs)

# 4) Precompute R̄ over the entire OOS period (1983-07 through 2010-12)
R_bar <- colMeans(X_all[test_idxs, , drop=FALSE])

# 5) Storage for log-likelihoods
ll_sample <- numeric(n_oos)
ll_glasso <- numeric(n_oos)
ll_lw     <- numeric(n_oos)

# 6) Rolling-window loop
window_end <- i_end
for(j in seq_len(n_oos)){
  t_idx <- window_end + 1
  Xw      <- X_all[(t_idx - (i_end - i_start + 1)):(t_idx - 1), ]
  R_next  <- as.numeric(X_all[t_idx, ])
  r_tilde <- R_next - R_bar      # <-- use fixed R̄
  
  # (a) sample-cov precision
  S_samp     <- cov(Xw)
  Omega_samp <- solve(S_samp)
  ld_samp    <- determinant(Omega_samp, logarithm=TRUE)$modulus
  ll_sample[j] <- ld_samp - (t(r_tilde) %*% Omega_samp %*% r_tilde)
  
  # (b) glasso precision
  S_gl       <- cov(Xw)
  gl         <- glasso(s=S_gl, rho=rho_optimal, penalize.diagonal=FALSE)
  Omega_g    <- gl$wi
  ld_g       <- determinant(Omega_g, logarithm=TRUE)$modulus
  ll_glasso[j] <- ld_g - (t(r_tilde) %*% Omega_g %*% r_tilde)
  
  # (c) Ledoit–Wolf precision
  Sigma_lw   <- covCor(Xw)
  Omega_lw   <- solve(Sigma_lw)
  ld_lw      <- determinant(Omega_lw, logarithm=TRUE)$modulus
  ll_lw[j]     <- ld_lw - (t(r_tilde) %*% Omega_lw %*% r_tilde)
  
  window_end <- window_end + 1
}

# 7) Summarize & print
avg_samp <- mean(ll_sample)
avg_gls  <- mean(ll_glasso)
avg_lw   <- mean(ll_lw)

cat("Average log-likelihoods (OOS):\n")
cat(sprintf(" Sample-cov: %.4f\n Glasso     : %.4f\n LW-shrink  : %.4f\n\n",
            avg_samp, avg_gls, avg_lw))
cat("Glasso minus:\n")
cat(sprintf(" Sample = %.4f\n LW     = %.4f\n",
            avg_gls - avg_samp,
            avg_gls - avg_lw))
