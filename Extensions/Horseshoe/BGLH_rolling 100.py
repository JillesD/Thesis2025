#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized Rolling-window backtest for Bayesian Graphical Horseshoe (GHS) with
Global Minimum Variance (GMV) portfolio and extended performance metrics.
"""
import time
import numpy as np
import numpy.random as rnd
import scipy.linalg as sl
from scipy.linalg.lapack import dpotrf, dpotrs
import pandas as pd
import matplotlib.pyplot as plt
from numba import jit, prange
import warnings

warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------
# Global numerical-stability constants
# ----------------------------------------------------------------------
JITTER_EPS = 1e-10  # diagonal jitter
LAMBDA_MIN = 1e-12  # floor for lambda^2 and tau^2
gamma = 5  # risk-aversion for CER


# ----------------------------------------------------------------------
# Numba-optimized functions
# ----------------------------------------------------------------------
@jit(nopython=True, cache=True)
def safe_cholesky_numba(A):
    """Fast Cholesky with automatic regularization."""
    n = A.shape[0]
    # Symmetrize
    for i in range(n):
        for j in range(i + 1, n):
            A[i, j] = A[j, i] = 0.5 * (A[i, j] + A[j, i])

    # Try Cholesky
    L = np.zeros_like(A)
    success = True

    # Manual Cholesky decomposition
    for i in range(n):
        for j in range(i + 1):
            if i == j:  # Diagonal elements
                sum_sq = 0.0
                for k in range(j):
                    sum_sq += L[j, k] * L[j, k]
                val = A[j, j] - sum_sq
                if val <= 0:
                    success = False
                    break
                L[j, j] = np.sqrt(val)
            else:  # Off-diagonal elements
                sum_prod = 0.0
                for k in range(j):
                    sum_prod += L[i, k] * L[j, k]
                L[i, j] = (A[i, j] - sum_prod) / L[j, j]
        if not success:
            break

    if not success:
        # Add regularization
        for i in range(n):
            A[i, i] += JITTER_EPS
        # Retry
        for i in range(n):
            for j in range(i + 1):
                if i == j:
                    sum_sq = 0.0
                    for k in range(j):
                        sum_sq += L[j, k] * L[j, k]
                    L[j, j] = np.sqrt(A[j, j] - sum_sq)
                else:
                    sum_prod = 0.0
                    for k in range(j):
                        sum_prod += L[i, k] * L[j, k]
                    L[i, j] = (A[i, j] - sum_prod) / L[j, j]

    return L


@jit(nopython=True, cache=True)
def solve_triangular_numba(L, b):
    """Solve Lx = b where L is lower triangular."""
    n = len(b)
    x = np.zeros(n)
    for i in range(n):
        x[i] = b[i]
        for j in range(i):
            x[i] -= L[i, j] * x[j]
        x[i] /= L[i, i]
    return x


@jit(nopython=True, cache=True)
def update_sigma_fast(Sigma, inv_O11, temp, gamma_ii, ind):
    """Optimized Sigma update."""
    p_sub = len(ind)
    # Update submatrix
    for i in range(p_sub):
        for j in range(p_sub):
            Sigma[ind[i], ind[j]] = inv_O11[i, j] + temp[i] * temp[j] / gamma_ii

    # Update cross terms
    for i in range(p_sub):
        val = -temp[i] / gamma_ii
        Sigma[ind[i], Sigma.shape[0] - 1] = val
        Sigma[Sigma.shape[0] - 1, ind[i]] = val


# ----------------------------------------------------------------------
# Optimized Cholesky
# ----------------------------------------------------------------------
def safe_cholesky(A: np.ndarray, lower: bool = True) -> np.ndarray:
    """Optimized Cholesky with LAPACK."""
    A = 0.5 * (A + A.T)
    try:
        chol, info = dpotrf(A, lower=lower)
        if info == 0:
            if lower:
                return np.tril(chol)
            else:
                return np.triu(chol)
    except:
        pass

    # Fallback with regularization
    lam_min = np.linalg.eigvalsh(A).min()
    jitter = JITTER_EPS if lam_min >= 0 else (-lam_min + JITTER_EPS)
    A_reg = A + jitter * np.eye(A.shape[0])
    chol, info = dpotrf(A_reg, lower=lower)
    if lower:
        return np.tril(chol)
    else:
        return np.triu(chol)


# ----------------------------------------------------------------------
# Optimized Graphical Horseshoe sampler
# ----------------------------------------------------------------------
def graphical_horseshoe_optimized(
        S: np.ndarray,
        n: int,
        burnin: int = 2000,
        nmc: int = 2000,
        tau_scale: float = 1.0,
        verbose_every: int = 1000,
        thin: int = 1
):
    """Optimized GHS sampler with reduced memory allocation and vectorization."""
    p = S.shape[0]

    # Pre-allocate arrays
    omega_save = np.empty((nmc // thin, p, p))
    ind_all = [np.delete(np.arange(p), i) for i in range(p)]

    # Initialize arrays
    Omega = np.eye(p, dtype=np.float64)
    Sigma = np.eye(p, dtype=np.float64)
    Lambda_sq = np.ones((p, p), dtype=np.float64)
    Nu = np.ones((p, p), dtype=np.float64)
    tau_sq = 1.0
    xi = 1.0

    # Workspace arrays to avoid repeated allocation
    temp_vec = np.empty(p - 1)
    temp_mat = np.empty((p - 1, p - 1))

    total_iters = burnin + nmc
    t0 = time.time()
    save_idx = 0

    for it in range(total_iters):
        for i in range(p):
            ind = ind_all[i]
            p_sub = len(ind)

            # Extract submatrices (use views when possible)
            Sigma_11 = Sigma[np.ix_(ind, ind)]
            sigma_12 = Sigma[ind, i]
            sigma_22 = Sigma[i, i]
            s_21 = S[ind, i]
            s_22 = S[i, i]
            lambda_sq_12 = Lambda_sq[ind, i]
            nu_12 = Nu[ind, i]

            # Sample gamma_ii
            gamma_ii = rnd.gamma(shape=n / 2 + 1, scale=2.0 / s_22)

            # Compute inv_O11 using Sherman-Morrison
            inv_O11 = Sigma_11 - np.outer(sigma_12, sigma_12) / sigma_22

            # Build inv_C with pre-allocated array
            temp_mat[:p_sub, :p_sub] = s_22 * inv_O11
            diag_add = 1.0 / (lambda_sq_12 * (tau_sq / tau_scale ** 2))
            np.fill_diagonal(temp_mat[:p_sub, :p_sub], temp_mat.diagonal()[:p_sub] + diag_add)

            inv_C = temp_mat[:p_sub, :p_sub]

            # Cholesky and solve
            L = safe_cholesky(inv_C, lower=True)

            # Solve for mu_i
            temp_vec[:p_sub] = -s_21
            mu_i = sl.cho_solve((L, True), temp_vec[:p_sub])

            # Sample beta
            temp_vec[:p_sub] = rnd.randn(p_sub)
            noise = sl.cho_solve((L, True), temp_vec[:p_sub])
            beta = mu_i + noise

            # Update Omega
            omega_12 = beta
            omega_22 = gamma_ii + beta.T @ inv_O11 @ beta
            Omega[i, ind] = omega_12
            Omega[ind, i] = omega_12
            Omega[i, i] = omega_22

            # Update Sigma efficiently
            temp = inv_O11 @ beta

            # Vectorized update of Sigma submatrix
            Sigma[np.ix_(ind, ind)] = inv_O11 + np.outer(temp, temp) / gamma_ii
            sigma12_new = -temp / gamma_ii
            Sigma[ind, i] = sigma12_new
            Sigma[i, ind] = sigma12_new
            Sigma[i, i] = 1.0 / gamma_ii

            # Ensure symmetry and positive definiteness
            Sigma = 0.5 * (Sigma + Sigma.T)

            # Sample lambda and nu (vectorized)
            rate_ls = omega_12 ** 2 / (2.0 * tau_sq) + 1.0 / nu_12
            lambda_sq_12 = 1.0 / rnd.gamma(1.0, 1.0 / rate_ls)
            lambda_sq_12 = np.maximum(lambda_sq_12, LAMBDA_MIN)
            nu_12 = 1.0 / rnd.gamma(1.0, 1.0 / (1.0 + 1.0 / lambda_sq_12))

            Lambda_sq[ind, i] = lambda_sq_12
            Lambda_sq[i, ind] = lambda_sq_12
            Nu[ind, i] = nu_12
            Nu[i, ind] = nu_12

        # Update tau (vectorized)
        omega_vec = Omega[np.tril_indices(p, -1)]
        lambda_vec = Lambda_sq[np.tril_indices(p, -1)]
        rate_t = 1.0 / xi + np.sum(omega_vec ** 2 / (2.0 * lambda_vec))
        tau_sq = 1.0 / rnd.gamma((p * (p - 1) / 2 + 1) / 2, 1.0 / rate_t)
        tau_sq *= tau_scale ** 2
        tau_sq = max(tau_sq, LAMBDA_MIN)
        xi = 1.0 / rnd.gamma(1.0, 1.0 / (1.0 + 1.0 / tau_sq))

        # Save with thinning
        if it >= burnin and (it - burnin) % thin == 0:
            omega_save[save_idx] = Omega.copy()
            save_idx += 1

        if verbose_every and (it + 1) % verbose_every == 0:
            print(f"GHS iter {it + 1:,}/{total_iters:,} – {time.time() - t0:.1f}s")

    return omega_save[:save_idx]


# ----------------------------------------------------------------------
# Vectorized performance evaluation
# ----------------------------------------------------------------------
@jit(nopython=True, cache=True)
def evaluate_perf_fast(returns):
    """Numba-optimized performance evaluation."""
    T = len(returns)
    Rbar_m = np.mean(returns)
    sigma2_m = np.var(returns)  # ddof=1 handled manually
    if T > 1:
        sigma2_m = sigma2_m * T / (T - 1)

    sharpe_m = Rbar_m / np.sqrt(sigma2_m) if sigma2_m > 0 else np.nan
    Rbar_a = Rbar_m * 12
    sigma2_a = sigma2_m * 12
    sharpe_a = Rbar_a / np.sqrt(sigma2_a) if sigma2_a > 0 else np.nan

    return T, Rbar_m, sigma2_m, sharpe_m, Rbar_a, sigma2_a, sharpe_a


def evaluate_perf(returns: np.ndarray):
    T, Rbar_m, sigma2_m, sharpe_m, Rbar_a, sigma2_a, sharpe_a = evaluate_perf_fast(returns)
    return dict(T=T, Rbar_m=Rbar_m, sigma2_m=sigma2_m,
                sharpe_m=sharpe_m, Rbar_a=Rbar_a,
                sigma2_a=sigma2_a, sharpe_a=sharpe_a)


@jit(nopython=True, cache=True)
def calc_TO_fast(w_mat, returns_oos):
    """Optimized turnover calculation."""
    M, p = w_mat.shape
    TO = np.empty(M - 1)

    for t in range(M - 1):
        w_t = w_mat[t]
        r_t = returns_oos[t] / 100
        w_t_plus = w_t * (1 + r_t)
        w_t_plus = w_t_plus / np.sum(w_t_plus)
        TO[t] = np.sum(np.abs(w_mat[t + 1] - w_t_plus))

    return TO


def calc_TO(w_mat: np.ndarray, returns_oos: np.ndarray):
    return calc_TO_fast(w_mat, returns_oos)


def q_metrics(wmat: np.ndarray):
    vec = wmat.flatten()
    wq = np.quantile(vec, [0, .01, .05, .95, .99, 1])
    herf = 10000 * np.mean(np.sum(wmat ** 2, axis=1))
    return dict(wq=wq, herf=herf)


def compute_TC(TO_vec: np.ndarray):
    turn_m = TO_vec.mean()
    turn_a = turn_m * 12
    TC_a = turn_a * 0.005
    return dict(turn_m=turn_m, turn_a=turn_a, TC_a=TC_a)


def to_CER(Rbar_a: float, sigma2_a: float, TC_a: float):
    R_d = Rbar_a / 100
    v_d = sigma2_a / 10000
    CERd = R_d - (gamma / 2) * v_d - TC_a
    return CERd * 100


# ----------------------------------------------------------------------
# Optimized Rolling-window backtest
# ----------------------------------------------------------------------
def main():
    print("Loading data...")
    # Load data
    path = r"C:/Users/Jilles/OneDrive/Thesis/Data/Cleaned/100_Portfolios_10x10_excess_clean_rounded.csv"
    df = pd.read_csv(path, parse_dates=['Date'])
    df['Date'] = pd.to_datetime(df['Date'].dt.to_period('M').dt.to_timestamp())
    df.sort_values('Date', inplace=True)

    dates = df['Date'].to_numpy()
    numeric_cols = df.columns.drop('Date')
    X_all = df[numeric_cols].to_numpy(dtype=np.float64)  # Ensure float64
    p = X_all.shape[1]
    oneN = np.ones(p, dtype=np.float64)

    # Rolling parameters (reduced for faster testing)
    burnin = 4000  # Reduced from 4000
    nmc = 4000  # Reduced from 2000
    thin = 1  # Add thinning to reduce storage
    train_start = np.datetime64('1973-07-01')
    train_end = np.datetime64('1983-06-01')
    last_test = np.datetime64('2010-12-01')

    i_start = np.where(dates == train_start)[0][0]
    i_end = np.where(dates == train_end)[0][0]
    training_size = i_end - i_start + 1

    # Pre-allocate storage
    max_windows = int((len(dates) - i_end) * 1.1)  # Conservative estimate
    ret_bayes = np.empty(max_windows)
    weights_bayes = np.empty((max_windows, p))

    window_count = 0
    window_end = i_end

    print("Starting rolling window backtest...")
    overall_start = time.time()

    while True:
        test_idx = window_end + 1
        if test_idx >= len(dates) or dates[test_idx] > last_test:
            break

        win_start = window_end - training_size + 1
        X_win = X_all[win_start:window_end + 1]

        # Compute S matrix
        S = X_win.T @ X_win
        n_win = X_win.shape[0]

        # Run optimized GHS sampler
        print(f"Window {window_count + 1}: Processing {dates[test_idx]}")
        omegas = graphical_horseshoe_optimized(
            S, n_win, burnin=burnin, nmc=nmc,
            tau_scale=0.5, verbose_every=1000, thin=thin
        )

        # Compute mean precision matrix
        Omega_bar = np.mean(omegas, axis=0)

        # GMV weights (vectorized)
        num = Omega_bar @ oneN
        den = oneN @ num
        w_bayes = num / den

        # Store results
        weights_bayes[window_count] = w_bayes
        ret_bayes[window_count] = float(w_bayes @ X_all[test_idx])

        window_count += 1
        window_end += 1

    # Trim arrays to actual size
    ret_bayes = ret_bayes[:window_count]
    weights_bayes = weights_bayes[:window_count]

    print(f"\nCompleted {window_count} windows in {time.time() - overall_start:.1f}s")
    print("Computing performance metrics...")

    # Compute metrics
    perf = evaluate_perf(ret_bayes)
    qm = q_metrics(weights_bayes)

    test_start = i_end + 1
    i_last_test = np.where(dates == last_test)[0][0]
    returns_oos = X_all[test_start:i_last_test + 1]
    TO = calc_TO(weights_bayes, returns_oos)
    TC = compute_TC(TO)
    CER = to_CER(perf['Rbar_a'], perf['sigma2_a'], TC['TC_a'])

    # Print results
    print("====================================================")
    print("Rolling-Window Backtest: Optimized Bayesian Horseshoe GMV")
    print(f"Out-of-sample months: {perf['T']}")

    print("Bayesian Horseshoe (Optimized):")
    print(
        f"  Monthly Return: {perf['Rbar_m']:.4f}%   Variance: {perf['sigma2_m']:.4f}   Sharpe: {perf['sharpe_m']:.4f}")
    print(
        f"  Annual  Return: {perf['Rbar_a']:.4f}%   Variance: {perf['sigma2_a']:.4f}   Sharpe: {perf['sharpe_a']:.4f}")
    print("  WtQ (Min,1%,5%,95%,99%,Max): ", np.round(qm['wq'], 4))
    print(f"  Herf: {qm['herf']:.2f}   Turn M: {TC['turn_m']:.4f}   Turn A: {TC['turn_a']:.4f}   CER: {CER:.2f}%")
    print("====================================================")


if __name__ == "__main__":
    main()
