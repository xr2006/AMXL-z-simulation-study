"""
estimate_amxl.py: step 3. Estimates AMXL-z (Sections 3.3.2 and 4.2.1) on one dataset. AMXL-z treats
each driver as an agent, with prior center m(z) = Gamma' phi(z):
    linear      AMXL-linear,    phi_L(z) = (1, inc, tod, dur)
    quadratic   AMXL-quadratic, phi_Q(z) = (1, inc, tod, dur, inc^2, dur^2, inc*tod, inc*dur, tod*dur)
z = (INCOME, TOD, DUR) is standardized with the training drivers. TOD^2 = TOD, so it is not repeated.

Decomposition algorithm:
    individual step   theta_n = argmin_b  lambda * sum_k W_k (b_k - m_k(z_n))^2 + NLL_n(b),  L <= b <= U
                      over the driver's C history choices; W_k = variance of attribute k; cost fixed at -1
    population step   Gamma = OLS of the theta_n (TT, POI) on phi(z_n)
    fixed point       the two steps alternate from the MNL-start tastes as intercept with zero slopes,
                      accelerated with SQUAREM, until a step changes m(z) by less than 0.01 %.

The heterogeneity that z does not explain is the empirical distribution of the training drivers'
deviations r_j = theta_j - m(z_j). A driver's tastes are distributed over the support {m(z_n) + r_j},
clipped to the bounds. This support predicts in the three settings of Section 4.2.2:
    predict_known   in-sample: training drivers, the support weighted by the likelihood of their history
                    (own r_n left out)
    predict_new     out-of-sample (no panel): test drivers from z alone, the logit averaged over the
                    support; taste m(z)
    predict_warm    out-of-sample (panel): test drivers, the support weighted by the likelihood of their
                    C history choices; taste = the weighted mean

lambda is selected from {1e-5, 1e-4, ..., 1e5} (11 values) by the NLL of the 400 development drivers'
evaluation choices, predicted from z alone.

Output: results/estimates/amxl_{linear,quadratic}/{dataset}_C{C}_{theta_all,fit,dev,best}.csv
    theta_all  theta_n of every training driver for every lambda
    fit        per lambda: Gamma, the weights W, fixed-point steps and failed drivers
    dev        per lambda: development NLL and ACC
    best       the selected lambda

Usage: python estimate_amxl.py --form quadratic --dataset noise20 --C 5
       (needs results/estimates/mnl_start/ from estimate_mnl_mxl.py)
"""
import argparse
import time
import warnings

import cvxpy as cp
import numpy as np
import pandas as pd

import config as cfg
import data_utils as du
from data_utils import BETA_NAME, J, with_cost, softmax_avail

warnings.filterwarnings('ignore')

L_BOUNDARY = [-2.5, -2.5, -1]          # TT, POI in [-2.5, 2.5] ($ per unit); CO fixed at -1
U_BOUNDARY = [2.5, 2.5, -1]
REGULARIZERS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5]
FORMS = ['linear', 'quadratic']
MODEL_NAME = {'linear': 'AMXL-linear', 'quadratic': 'AMXL-quadratic'}
OUT_DIR = {f: cfg.EST_DIR / f'amxl_{f}' for f in FORMS}
TERMS = du.TERMS


# ============================================================ individual step
def driver_problems(w):
    """Per-driver data of the MAP problem, in iid order: M (n, C*J, K), the attributes with one row per
    (occasion, alternative), and c (n, K), the summed attributes of the chosen alternatives."""
    X, y, A, drv = du.arrays(w)
    assert (A == 1).all(), 'the vectorized MAP assumes that every alternative is available'
    ids = np.unique(drv)
    n, (K, J_) = len(ids), X.shape[1:]
    C = len(drv) // n
    assert len(drv) == n * C and (drv.reshape(n, C) == ids[:, None]).all(), 'expects C sorted rows per driver'
    Xd, yd = X.reshape(n, C, K, J_), y.reshape(n, C)
    M = Xd.transpose(0, 1, 3, 2).reshape(n, C * J_, K)
    c = Xd[np.arange(n)[:, None], np.arange(C)[None, :], :, yd].sum(1)
    return M, c


class DriverMAP:
    """min_b  lambda * sum_k W_k (b_k - center_k)^2 + sum_t [log sum_j exp(x_tj'b) - x_t,y_t'b],
       s.t.   L <= b <= U.
    The data enter as cvxpy parameters, so the problem is compiled once and every driver only resets
    the parameter values."""

    def __init__(self, C):
        K = len(BETA_NAME)
        self.b = cp.Variable(K)
        self.M = cp.Parameter((C * J, K))
        self.c = cp.Parameter(K)
        self.s = cp.Parameter(K, nonneg=True)       # sqrt(lambda * W)
        self.r = cp.Parameter(K)                    # s * center
        V = cp.reshape(self.M @ self.b, (C, J), order='C')
        nll = cp.sum(cp.log_sum_exp(V, axis=1)) - self.c @ self.b
        penalty = cp.sum_squares(cp.multiply(self.s, self.b) - self.r)
        cons = [self.b >= np.asarray(L_BOUNDARY, float), self.b <= np.asarray(U_BOUNDARY, float)]
        self.problem = cp.Problem(cp.Minimize(penalty + nll), cons)

    def solve(self, M, c, center, lam, W):
        """The driver's MAP and whether a solver succeeded (if none does: the prior center)."""
        self.M.value, self.c.value = M, c
        s = np.sqrt(lam * np.asarray(W, float))
        self.s.value, self.r.value = s, s * np.asarray(center, float)
        for solver in [cp.CLARABEL, cp.MOSEK, cp.ECOS, cp.SCS]:
            try:
                self.problem.solve(solver=solver, verbose=False)
                if self.problem.status in ['optimal', 'optimal_inaccurate']:
                    return self.b.value.copy(), True
            except Exception:
                pass
        return np.asarray(center, float).copy(), False


def individual_step(solver, drivers, centers, lam, W):
    M, c = drivers
    theta = np.zeros((len(M), centers.shape[1]))
    n_failed = 0
    for i in range(len(M)):
        theta[i], ok = solver.solve(M[i], c[i], centers[i], lam, W)
        n_failed += not ok
    return theta, n_failed


# ============================================================ fixed point over Gamma (SQUAREM)
def fixed_point(solver, drivers, Phi, gamma_0, lam, W, tol=1e-4, max_cycles=50):
    """Gamma = step(Gamma), a step being the individual step around m(z_n) followed by the OLS population
    step. Accelerated with SQUAREM (Varadhan and Roland, 2008). Stops when a step changes m(z) by less
    than tol (relative). Returns Gamma, the theta_n at the prior center Phi @ Gamma, the number of
    failed drivers and the number of steps."""
    shape = gamma_0.shape

    def step(x):
        theta, n = individual_step(solver, drivers, with_cost(Phi @ x.reshape(shape)), lam, W)
        return np.linalg.lstsq(Phi, theta[:, :2], rcond=None)[0].ravel(), theta, n

    def change(x_new, x):
        return np.linalg.norm(Phi @ (x_new - x).reshape(shape)) / np.linalg.norm(Phi @ x.reshape(shape))

    x, n_steps = gamma_0.ravel(), 0
    for _ in range(max_cycles):
        x1, _, _ = step(x)
        x2, _, _ = step(x1)
        r, v = x1 - x, x2 - 2 * x1 + x
        alpha = min(-np.linalg.norm(r) / max(np.linalg.norm(v), 1e-12), -1.0)    # -1: x_ext = x2
        x_ext = x - 2 * alpha * r + alpha ** 2 * v
        x_new, theta, n = step(x_ext)
        n_steps += 3
        res = change(x_new, x_ext)
        if res > change(x2, x1):                   # extrapolation did not help: plain double step
            x_ext = x2
            x_new, theta, n = step(x2)
            n_steps += 1
            res = change(x_new, x2)
        if res < tol:
            break
        x = x_new
    return x_ext.reshape(shape), theta, n, n_steps


# ============================================================ prediction from the empirical distribution
def empirical_probs(m, r, fut, hist=None, leave_out=False, chunk=500):
    """Tastes of each driver distributed over {m(z_n) + r_j} (m (drivers, 2) in iid order, r (S, 2) the
    training deviations), clipped to the bounds. Returns the choice probabilities of the rows of fut
    and the tastes (drivers, 2). Without a history, the support is averaged uniformly and the taste is
    m(z). With the history rows, the support is weighted by the likelihood of the driver's history.
    With leave_out, the drivers are the training drivers, and each driver's own r_n is left out."""
    B = with_cost(np.clip(m[:, None, :] + r[None], L_BOUNDARY[:2], U_BOUNDARY[:2]))     # (drivers, S, K)
    loglik = np.zeros(B.shape[:2])
    if hist is not None:
        Xh, yh, _, dh = du.arrays(hist)
        for ch in np.array_split(np.arange(len(yh)), max(1, len(yh) // chunk)):
            V = np.einsum('nsk,nkj->nsj', B[dh[ch]], Xh[ch])
            V = V - V.max(-1, keepdims=True)
            np.add.at(loglik, dh[ch], V[np.arange(len(ch)), :, yh[ch]] - np.log(np.exp(V).sum(-1)))
        if leave_out:
            loglik[np.arange(len(loglik)), np.arange(len(loglik))] = -np.inf
    w = np.exp(loglik - loglik.max(1, keepdims=True))
    w /= w.sum(1, keepdims=True)
    X, _, A, drv = du.arrays(fut)
    P = np.empty((len(drv), X.shape[2]))
    for ch in np.array_split(np.arange(len(drv)), max(1, len(drv) // chunk)):
        Pr = softmax_avail(np.einsum('nsk,nkj->nsj', B[drv[ch]], X[ch]), A[ch][:, None, :])
        P[ch] = np.einsum('ns,nsj->nj', w[drv[ch]], Pr)
    return P, (m if hist is None else np.einsum('ns,nsk->nk', w, B[:, :, :2]))


# ============================================================ one full run
def run(dataset, C, form):
    tag, out, terms = cfg.tag(dataset, C), OUT_DIR[form], TERMS[form]
    out.mkdir(parents=True, exist_ok=True)
    w_tr = du.history(du.load_split('train', dataset), C)
    w_dv = du.future(du.load_split('dev', dataset))
    print(f"{MODEL_NAME[form]} [{tag}]: train {w_tr['iid'].nunique()} drivers x {C} choices, "
          f"dev {w_dv['iid'].nunique()} drivers x {cfg.H} evaluation choices")

    # weights W_k of the penalty: variance of each attribute over the training choices
    W = np.maximum([w_tr[[f'ALT{j}_{a}' for j in range(J)]].values.ravel().var(ddof=1) for a in BETA_NAME], 1e-6)

    # starting value: MNL without z as intercept, zero slopes
    start = cfg.EST_DIR / 'mnl_start' / f'mnl_start_{tag}.csv'
    if not start.exists():
        raise FileNotFoundError(f'{start} not found: run estimate_mnl_mxl.py --model MNL-start first')
    beta_0 = pd.read_csv(start).iloc[0][BETA_NAME].values.astype(float)
    gamma_0 = np.zeros((len(terms), 2))
    gamma_0[0] = beta_0[:2]

    drivers = driver_problems(w_tr)
    solver = DriverMAP(C)
    ids = np.unique(w_tr['iid'])
    F_tr = du.driver_z(w_tr)
    Phi_tr, Phi_dv = du.basis(F_tr, F_tr, form), du.basis(du.driver_z(w_dv), F_tr, form)
    y_dv = w_dv['CHOICE'].values

    thetas, fits, dev = [], [], []
    for lam in REGULARIZERS:
        t0 = time.time()
        gamma, theta, n_failed, n_steps = fixed_point(solver, drivers, Phi_tr, gamma_0, lam, W)
        thetas.append(pd.DataFrame(theta, columns=BETA_NAME).assign(regularizer=lam, iid=ids))
        fits.append({'regularizer': lam,
                     **{f'gamma_{t}_{k}': gamma[i, j] for i, t in enumerate(terms) for j, k in enumerate(BETA_NAME[:2])},
                     **{f'W_{k}': v for k, v in zip(BETA_NAME, W)},
                     'n_steps': n_steps, 'n_failed': n_failed})
        r = theta[:, :2] - Phi_tr @ gamma
        nll, acc = du.nll_acc(empirical_probs(Phi_dv @ gamma, r, w_dv)[0], y_dv)
        dev.append({'regularizer': lam, 'NLL_dev': nll, 'ACC_dev': acc})
        print(f'  lambda={lam:.0e}: {n_steps:2d} steps, {n_failed} failed drivers, {time.time() - t0:5.1f} s | '
              f'dev NLL {nll:.4f}, ACC {acc:.4f}', flush=True)

    dev = pd.DataFrame(dev)
    best = dev.loc[[dev['NLL_dev'].idxmin()]]
    pd.concat(thetas, ignore_index=True).to_csv(out / f'{tag}_theta_all.csv', index=False)
    pd.DataFrame(fits).to_csv(out / f'{tag}_fit.csv', index=False)
    dev.to_csv(out / f'{tag}_dev.csv', index=False)
    best.to_csv(out / f'{tag}_best.csv', index=False)
    print(f"  selected lambda = {best.iloc[0]['regularizer']:.0e} (dev NLL {best.iloc[0]['NLL_dev']:.4f}) -> {out}")


# ============================================================ prediction from the saved fit
def deviations(dataset, C, form):
    """Gamma, the empirical deviations r_j = theta_j - m(z_j) of the training drivers at the selected lambda,
    and the training drivers' raw z (for the standardization of phi)."""
    t, out = cfg.tag(dataset, C), OUT_DIR[form]
    lam = pd.read_csv(out / f'{t}_best.csv').iloc[0]['regularizer']
    th = pd.read_csv(out / f'{t}_theta_all.csv')
    theta = th[np.isclose(th['regularizer'], lam)].sort_values('iid')[BETA_NAME].values
    fit = pd.read_csv(out / f'{t}_fit.csv')
    fit = fit[np.isclose(fit['regularizer'], lam)].iloc[0]
    gamma = np.array([[fit[f'gamma_{term}_{k}'] for k in BETA_NAME[:2]] for term in TERMS[form]])
    F_tr = du.driver_z(du.history(du.load_split('train', dataset), C))
    return gamma, theta[:, :2] - du.basis(F_tr, F_tr, form) @ gamma, F_tr


def predict_known(dataset, C, w_hist, w_fut, form):
    gamma, r, F_tr = deviations(dataset, C, form)
    return empirical_probs(du.basis(du.driver_z(w_fut), F_tr, form) @ gamma, r, w_fut, w_hist, leave_out=True)


def predict_new(dataset, C, w_fut, form):
    gamma, r, F_tr = deviations(dataset, C, form)
    return empirical_probs(du.basis(du.driver_z(w_fut), F_tr, form) @ gamma, r, w_fut)


def predict_warm(dataset, C, w_hist, w_fut, form):
    gamma, r, F_tr = deviations(dataset, C, form)
    return empirical_probs(du.basis(du.driver_z(w_fut), F_tr, form) @ gamma, r, w_fut, w_hist)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='AMXL-linear / AMXL-quadratic on one dataset')
    ap.add_argument('--form', choices=FORMS, required=True)
    ap.add_argument('--dataset', choices=list(cfg.NOISE), required=True)
    ap.add_argument('--C', type=int, choices=cfg.C_LIST, required=True)
    a = ap.parse_args()
    run(a.dataset, a.C, a.form)
    cfg.record_cost(MODEL_NAME[a.form], a.dataset, a.C, n_fits=len(REGULARIZERS))
