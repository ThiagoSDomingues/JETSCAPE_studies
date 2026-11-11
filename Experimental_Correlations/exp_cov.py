"""Non-diagonal experimental covariance for the JETSCAPE soft-sector Bayesian analysis.

Model for one observable measured in centrality bins c_i (fractions in [0,1]):
    y_i = f(c_i) + eps_stat_i + eps_unc_i + eps_cor_i
    f ~ GP(0, A^2 exp(-(dc)^2 / L^2))            smooth physical trend
    Cov_sys_ij = s_i s_j [(1-alpha) delta_ij + alpha exp(-(dc)^2 / ell^2)]
alpha = correlated fraction of the systematic variance, ell = its correlation length.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize


def rbf(c1, c2, length):
    return np.exp(-np.subtract.outer(c1, c2) ** 2 / length ** 2)


def sys_cov(c, sys, alpha, ell):
    """Systematic covariance of one observable."""
    return np.outer(sys, sys) * ((1 - alpha) * np.eye(len(c)) + alpha * rbf(c, c, ell))


def neg_log_marglike(params, c, y, stat, sys, ell=None):
    """GP marginal likelihood of the data; params = (logA, logL, logit alpha[, log ell])."""
    A, L = np.exp(params[0]), np.exp(params[1])
    alpha = 1 / (1 + np.exp(-params[2]))
    ell = np.exp(params[3]) if ell is None else ell
    K = A ** 2 * rbf(c, c, L) + np.diag(stat ** 2) + sys_cov(c, sys, alpha, ell)
    K += 1e-10 * np.trace(K) / len(c) * np.eye(len(c))
    try:
        cf = cho_factor(K, lower=True)
    except np.linalg.LinAlgError:
        return 1e30
    r = y - y.mean()
    return 0.5 * r @ cho_solve(cf, r) + np.log(np.diag(cf[0])).sum()


def fit_observable(c, y, stat, sys, ell=None):
    """Best-fit (alpha, ell). If ell is given it is held fixed (recommended: ell is
    degenerate with the trend f, so scan it or give it a prior instead of fitting)."""
    x0 = [np.log(y.std() + 1e-12), np.log(0.5), 0.0] + ([] if ell is not None else [np.log(0.5)])
    best = min((minimize(neg_log_marglike, np.array(x0) + d, args=(c, y, stat, sys, ell),
                         method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-6, maxiter=4000))
                for d in (0.0, 0.7, -0.7)), key=lambda r: r.fun)
    alpha = 1 / (1 + np.exp(-best.x[2]))
    return dict(alpha=alpha, ell=ell if ell is not None else np.exp(best.x[3]),
                trend_length=np.exp(best.x[1]), nll=best.fun)


def build_cov(obs, groups, alpha, ell, rho0):
    """Full covariance for a list of observables.

    obs    : dict name -> dict(c=..., stat=..., sys=...)   (order defines the data vector)
    groups : dict name -> group label (as expt_obs_corr_group in bayes_mcmc.py)
    alpha  : float or dict name -> correlated fraction
    ell    : float or dict group -> correlation length
    rho0   : correlation between different observables of the same group
    Setting alpha=0 gives the diagonal matrix; alpha=1, ell=0.5, rho0=0.8 with stat
    folded into sys reproduces the JETSCAPE App. D ansatz.
    """
    names = list(obs)
    sizes = [len(obs[n]["c"]) for n in names]
    off = np.concatenate([[0], np.cumsum(sizes)])
    cov = np.zeros((off[-1], off[-1]))
    get = lambda v, k: v[k] if isinstance(v, dict) else v
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if groups[a] != groups[b]:
                continue
            oa, ob = obs[a], obs[b]
            sa = np.sqrt(get(alpha, a)) * oa["sys"]
            sb = np.sqrt(get(alpha, b)) * ob["sys"]
            blk = np.outer(sa, sb) * rbf(oa["c"], ob["c"], get(ell, groups[a]))
            if a == b:
                blk = blk + np.diag(oa["stat"] ** 2 + (1 - get(alpha, a)) * oa["sys"] ** 2)
            else:
                blk = blk * rho0
            cov[off[i]:off[i + 1], off[j]:off[j + 1]] = blk
    return cov


if __name__ == "__main__":  # self-test on synthetic data
    rng = np.random.default_rng(1)
    c = np.linspace(0.025, 0.65, 14)
    truth = 0.03 + 0.12 * c - 0.08 * c ** 2
    stat, sys = 0.002 * truth, 0.05 * truth
    for a_true in (0.0, 0.9):
        est = []
        for _ in range(40):
            y = truth + rng.normal(0, stat) + rng.multivariate_normal(np.zeros(14), sys_cov(c, sys, a_true, 0.5))
            est.append(fit_observable(c, y, stat, sys, ell=0.5)["alpha"])
        print(f"alpha_true={a_true}: median fitted alpha={np.median(est):.2f}")
    obs = {"v22": dict(c=c, stat=stat, sys=sys), "v32": dict(c=c, stat=stat / 2, sys=sys / 2),
           "mean_pT_pion": dict(c=c, stat=stat, sys=sys)}
    grp = {"v22": "flows", "v32": "flows", "mean_pT_pion": "mean_pT"}
    S = build_cov(obs, grp, 0.9, 0.5, 0.8)
    print("shape", S.shape, "min eig", np.linalg.eigvalsh(S).min() > 0, "symmetric", np.allclose(S, S.T))
