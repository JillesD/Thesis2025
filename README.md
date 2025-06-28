# Bachelor Thesis – Mean-Variance Optimization with Graphical Estimators

This repository contains all code used for my bachelor thesis in Econometrics at Erasmus University Rotterdam. The thesis replicates the results of Goto and Xu (2021) on mean-variance portfolio optimization and extends their methodological framework using recent advancements in precision matrix estimation.

## Structure

The repository is organized into the following folders:

- **`data/`**  
  Contains all datasets used throughout the research, including the 48 Industry Portfolios and the 100 Fama-French Sorted Portfolios.

- **`descriptives/`**  
  Includes supporting scripts for exploratory and diagnostic analysis:  
  - Condition numbers of covariance and precision matrices  
  - Optimal penalty parameters (λ and ρ) for the graphical elastic net and graphical lasso

- **`replicate/`**  
  Contains the full replication of Goto and Xu’s methodology across all datasets, including covariance estimation, precision matrix construction, and portfolio performance evaluation.

- **`extensions/`**  
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

Goto, S., & Xu, Y. (2021). *Improving Mean-Variance Optimization through Shrinkage and Graphical Models*. Journal of Financial Econometrics. [Add DOI or link]

---

If you use or adapt the code in this repository, please give credit or cite appropriately. For questions or clarifications, feel free to reach out.
