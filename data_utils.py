"""
data_utils.py: helpers shared by the estimation and evaluation scripts. They load a split, select the
history or evaluation occasions, convert to arrays, and build the basis phi(z) of the driver
characteristics.
"""
import numpy as np
import pandas as pd

import config as cfg

BETA_NAME = ['TT', 'POI', 'CO']              # attribute order of the arrays; the cost coefficient is -1
INDIV_ATTR = ['INCOME', 'TOD', 'DUR']        # driver characteristics z
J = cfg.J

# phi(z) of the z-based models. 'linear': MNL, MXL and AMXL-linear; 'quadratic': AMXL-quadratic.
# TOD is a dummy, so TOD^2 = TOD is left out of the quadratic.
TERMS = {'linear': ['1', 'inc', 'tod', 'dur'],
         'quadratic': ['1', 'inc', 'tod', 'dur', 'inc2', 'dur2', 'inc_tod', 'inc_dur', 'tod_dur']}


def load_split(split, dataset):
    """Wide panel of one split: one row per choice occasion."""
    return pd.read_csv(cfg.DATA_DIR / f'{split}_{dataset}.csv')


def load_truth(split, dataset):
    """True tastes of the drivers of one split, in iid order."""
    truth = pd.read_csv(cfg.DATA_DIR / f'{split}_{dataset}_truebeta.csv')
    assert (truth['iid'].values == np.arange(len(truth))).all()
    return truth


def history(w, C):
    """The first C occasions of every driver (estimation / the panel used for prediction)."""
    return w[w['t'] <= C].reset_index(drop=True)


def future(w):
    """The evaluation occasions t = 21..25 of every driver."""
    return w[w['t'] > cfg.T_HIST].reset_index(drop=True)


def arrays(w):
    """X (n, K, J) in BETA_NAME order, chosen alternative y (n,), availability A (n, J) and the
    driver index iid (n,) of every choice occasion."""
    X = np.stack([w[[f'ALT{j}_{a}' for j in range(J)]].values for a in BETA_NAME], axis=1).astype(float)
    return X, w['CHOICE'].values, w[[f'ALT{j}_AV' for j in range(J)]].values, w['iid'].values


def driver_z(w):
    """Raw (INCOME, TOD, DUR) of every driver, in iid order. The drivers of a split are numbered 0..n-1,
    so a driver's iid is also its row in every per-driver array."""
    z = w.groupby('iid')[INDIV_ATTR].first().sort_index()
    assert (z.index.values == np.arange(len(z))).all(), 'expects drivers numbered 0..n-1'
    return z.values.astype(float)


def poly(Z, form):
    """phi(z) of standardized (INCOME, TOD, DUR), columns in TERMS[form] order."""
    inc, tod, dur = Z.T
    terms = [np.ones(len(Z)), inc, tod, dur]
    if form == 'quadratic':
        terms += [inc ** 2, dur ** 2, inc * tod, inc * dur, tod * dur]
    return np.column_stack(terms)


def basis(F, F_train, form):
    """phi(z) of the drivers F (raw z), standardized by the training drivers' mean and SD."""
    return poly((F - F_train.mean(0)) / F_train.std(0), form)


def with_cost(b):
    """Append the fixed cost coefficient -1: (..., 2) tastes -> (..., 3) in BETA_NAME order."""
    return np.concatenate([b, -np.ones(b.shape[:-1] + (1,))], axis=-1)


def softmax_avail(V, A):
    """Softmax over the last axis, restricted to the available alternatives (A == 1)."""
    V_masked = np.where(A == 1, V, -1e10)
    exp_V = np.where(A == 1, np.exp(V_masked - V_masked.max(axis=-1, keepdims=True)), 0.0)
    return exp_V / exp_V.sum(axis=-1, keepdims=True)


def mixture_probs(betas, X, A):
    """betas (n, m, K): m taste vectors per occasion; X (n, K, J); A (n, J).
    Returns the choice probabilities (n, J) averaged over the m taste vectors."""
    V = np.einsum('nmk,nkj->nmj', betas, X)
    return softmax_avail(V, A[:, None, :]).mean(axis=1)


def nll_acc(P, y):
    """NLL per choice divided by J (Eq. 16) and accuracy (Eq. 17)."""
    nll = -np.log(np.maximum(P[np.arange(len(y)), y], 1e-300)).mean() / P.shape[1]
    return nll, (P.argmax(1) == y).mean()
