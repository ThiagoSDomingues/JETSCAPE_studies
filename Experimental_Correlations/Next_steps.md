Six studies produce every plot in the talk. Three figures need no new work, and the backup figures come for free from the same chains.

## Studies, in order

| # | Study | Output | Done by |
|---|---|---|---|
| S0 | Get Nara's chains, plots and setup; run the summer school notebook with real data, diagonal covariance | Baseline posterior that matches the notebook's | 13 Nov |
| S1 | Load HEPData with stat and sys kept separate; build the covariance with `build_cov`; check it reproduces the paper's ansatz (`compute_cov`) and is positive definite | One function that returns Σ_exp for any model | 15 Nov |
| S2 | MCMC runs, changing only Σ_exp: A diagonal; B compound symmetry (suggest ρ = 0.3, 0.6, 0.9); C RBF (suggest ℓ = 0.2, 0.5, 1.0) | 7 converged chains | 20 Nov |
| S3 | Post-process S2: viscosity bands, shift of each parameter in units of the diagonal σ, 90% widths | Result I and II figures | 22 Nov |
| S4 | GP on the data: smooth fit per observable, pulls, fit of the correlated fraction α per group (ℓ fixed, scanned) | Data-driven figures | 25 Nov |
| S5 | Run D: one MCMC with the covariance from S4 | D band added to the viscosity figures | 27 Nov |

Results freeze on 29 Nov; the last week is slides and rehearsal only. A closure test (pseudo-data with known correlated noise, analysed with A and D) is optional and only if S5 finishes early.

## Plot to study

**Main slides**

| Figure file | Slide | Comes from |
|---|---|---|
| `jetscape_fig23_viscous_posterior.pdf` | A known warning | Taken from the paper (no study) |
| `eta_s_posterior_runs.pdf` | Result I | S2 + S3 (D from S5) |
| `zeta_s_posterior_runs.pdf` | Result I | S2 + S3 (D from S5) |
| `posterior_shift_cs_rbf.pdf` | Result II | Nara's plot (S0), or S3 if her setup differs |
| `fig_w_short.pdf` | Case study | Already done |
| `gp_fit_data.pdf` | GP on the measurements | S4 |
| `pull_distribution.pdf` | First step | S4 |
| `alpha_per_group.pdf` | First step | S4 |

**Backup slides**

| Figure file | Comes from |
|---|---|
| `corr_matrix_ansatz.pdf`, `corr_matrix_gp.pdf` | S1 and S4 (heatmaps of matrices you already built) |
| `parameter_widths_runs.pdf` | S3 |
| `corner_diag_vs_corr.pdf` | S2 chains |
| `posterior_predictive_diag.pdf`, `posterior_predictive_corr.pdf` | S2 chains |
| `closure_coverage.pdf` | Optional closure test |
| `kernel_comparison.pdf` | S3 (compound symmetry vs RBF is already this comparison) |

## What blocks what

- S0 decides whether Result II is Nara's figure or yours. Ask her first.
- S1 blocks everything else.
- S2 is the only compute-heavy step; start it the moment S1 passes its checks.
- S4 does not depend on S2, so it can run in parallel while the chains sample.

## Rules to fix before running

- **Convergence:** same number of walkers and steps for every run, and two independent chains for run A, agreed before looking at results.
- **Minimum viable talk:** S0 to S3 plus the case study. If time runs out, S4 and S5 become one outlook sentence and the slides "GP on the measurements" and "First step" move to backup.
