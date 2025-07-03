# Bachelor Thesis – Mean-Variance Optimization with Graphical Estimators

This repository contains all code used for my bachelor thesis in Econometrics at Erasmus University Rotterdam. The thesis replicates the results of Goto and Xu (2021) on mean-variance portfolio optimization and extends their methodological framework using recent advancements in precision matrix estimation.

## Structure

The repository is organized into the following folders:

- **`Data/`**  
  Contains all datasets used throughout the research, including the 48 Industry Portfolios and the 100 Fama-French Sorted Portfolios.

- **`Descriptives/`**  
  Includes supporting scripts for exploratory and diagnostic analysis:  
  - Condition numbers of covariance and precision matrices  
  - Optimal penalty parameters (λ and ρ) for the graphical elastic net and graphical lasso

- **`Replicate/`**  
  Contains the full replication of Goto and Xu’s methodology across all datasets, including covariance estimation, precision matrix construction, and portfolio performance evaluation. The subfolder likelihood contains the code for the replication of Table 6 in Goto, S., & Xu, Y. (2021). Improving Mean-Variance Optimization through Shrinkage and Graphical Models. Journal of Financial Econometrics.

- **`Extensions/`**  
  Implements novel extensions beyond the original paper:
  - `graphical_elastic_net/`: Code for estimating precision matrices using the graphical elastic net
  - `graphical_horseshoe/`: Code for the Bayesian graphical horseshoe estimator and corresponding backtests

## Thesis Summary

The goal of this thesis is twofold:
1. **Replication** – Accurately reproduce the empirical findings of Goto and Xu (2021), who study the role of precision matrix estimation in global minimum variance (GMV) portfolio optimization.
2. **Extension** – Improve upon their framework by introducing more advanced estimators:  
   - The **graphical elastic net**, which balances sparsity and stability  
   - The **graphical horseshoe**, a fully Bayesian shrinkage method designed for high-dimensional settings

## Citation

Please cite the original work if using this code for research purposes:

Goto, S., & Xu, Y. (2021). *Improving Mean-Variance Optimization through Shrinkage and Graphical Models*. Journal of Financial Econometrics.
Kovács, S., Ruckstuhl, T., Obrist, H., & Bühlmann, P. (2021, January 6). Graphical Elastic Net and Target Matrices: Fast Algorithms and Software for Sparse Precision Matrix Estimation.
Ledoit, O. and Wolf, M. (2004a). Honey, I shrunk the sample covariance matrix. Journal of Portfolio Management, 30(4):110–119.
Li, Y., Craig, B. A., and Bhadra, A. (2019). The graphical horseshoe estimator for inverse covariance matrices. arXiv preprint arXiv:1707.06661.

---

If you use or adapt the code in this repository, please give credit or cite appropriately. For questions or clarifications, feel free to reach out.
