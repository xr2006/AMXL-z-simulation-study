"""
estimate_mnl_mxl.py: step 2. Estimates the MNL and MXL benchmarks of Section 4.2.1 with xlogit, on the
C history choices of the 1,200 training drivers of one dataset.

The models use the utility of the data-generating process, so the cost enters with its coefficient
fixed at -1 (xlogit's `addit`) and there are no ASCs. Each taste interacts with the three z variables,
standardized with the training drivers:
    beta_k(z) = gamma_k' (1, inc, tod, dur),   k in {TT, POI}.

    MNL        fixed tastes beta_k(z); the choices are treated as independent.
    MXL        beta_k,n = gamma_k' (1, inc, tod, dur) + eta_k,n with eta_k,n ~ N(0, omega_k^2), constant
               over a driver's choices (panel mixed logit). Simulated with 600 Halton draws per driver.
    MNL-start  MNL without z (only beta_TT, beta_POI). Not reported in the paper. It is the starting
               value of the population-level tastes of AMXL (Section 4.2.1).

None of these models has hyperparameters, so the development drivers are not used.

Output: results/estimates/{mnl,mxl,mnl_start}/{model}_{dataset}_C{C}.csv (coefficients, LL, convergence
flag) and *_summary_*.txt (the xlogit summary). With a single --model, the CPU time and peak memory
of the process go to results/compute_cost/ (Table 4).

Usage:
    python estimate_mnl_mxl.py --model MXL --dataset noise20 --C 5
    python estimate_mnl_mxl.py --all          # every model and dataset in one process (no cost record)
"""
import argparse
import contextlib

import pandas as pd
from xlogit import MixedLogit, MultinomialLogit

import config as cfg
import data_utils as du

TASTES = cfg.TASTES
N_DRAWS = 600
MODELS = ['MNL-start', 'MNL', 'MXL']
FOLDER = {'MNL-start': 'mnl_start', 'MNL': 'mnl', 'MXL': 'mxl'}
TERMS = du.TERMS['linear']
ZVAR = {(k, t): k if t == '1' else f'{k}_{t}' for k in TASTES for t in TERMS}     # taste x phi(z)


def out_path(model, dataset, C):
    return cfg.EST_DIR / FOLDER[model] / f'{FOLDER[model]}_{cfg.tag(dataset, C)}.csv'


def load_long(dataset, C):
    """History choices t <= C of the training drivers in long format (one row per alternative), with the
    interactions of TT and POI with phi(z)."""
    wide = du.history(du.load_split('train', dataset), C)
    Phi = du.basis(wide[du.INDIV_ATTR].values.astype(float), du.driver_z(wide), 'linear')
    pieces = []
    for j in range(cfg.J):
        piece = pd.DataFrame({
            'iid': wide['iid'], 'choice_id': wide['choice_id'], 'alt': f'ALT{j}',
            'TT': wide[f'ALT{j}_TT'], 'POI': wide[f'ALT{j}_POI'], 'CO': wide[f'ALT{j}_CO'],
            'AV': wide[f'ALT{j}_AV'], 'chosen': (wide['CHOICE'] == j).astype(int)})
        for (k, t), v in ZVAR.items():
            if t != '1':
                piece[v] = piece[k].values * Phi[:, TERMS.index(t)]
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True).sort_values(['iid', 'choice_id', 'alt']).reset_index(drop=True)


def fit(model, data):
    varnames = TASTES if model == 'MNL-start' else list(ZVAR.values())
    args = dict(X=data[varnames], y=data['chosen'], varnames=varnames, alts=data['alt'],
                ids=data['choice_id'], avail=data['AV'], addit=(-data['CO']).values, verbose=0)
    if model == 'MXL':                  # the intercept terms of the tastes are random (normal), per driver
        m = MixedLogit()
        m.fit(**args, randvars={v: 'n' for v in TASTES}, panels=data['iid'], n_draws=N_DRAWS,
              halton=True, random_state=0)
    else:
        m = MultinomialLogit()
        m.fit(**args)
    return m


def run(model, dataset, C):
    path = out_path(model, dataset, C)
    path.parent.mkdir(parents=True, exist_ok=True)
    m = fit(model, load_long(dataset, C))
    c = dict(zip(m.coeff_names, m.coeff_))
    if model == 'MNL-start':
        row = {v: c[v] for v in TASTES}
    else:
        row = {f'gamma_{t}_{k}': c[v] for (k, t), v in ZVAR.items()}
    if model == 'MXL':
        row.update({f'sd_{v}': abs(c[f'sd.{v}']) for v in TASTES})       # xlogit may report a negative sd
    row.update({'CO': -1.0, 'LL_train': m.loglikelihood, 'converged': m.convergence})
    pd.DataFrame([row]).to_csv(path, index=False)
    with open(str(path).replace('.csv', '_summary.txt'), 'w') as f, contextlib.redirect_stdout(f):
        m.summary()
    print(f'[{model} {cfg.tag(dataset, C)}] converged={m.convergence}  LL={m.loglikelihood:.2f}  -> {path.name}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--model', choices=MODELS)
    ap.add_argument('--dataset', choices=list(cfg.NOISE))
    ap.add_argument('--C', type=int, choices=cfg.C_LIST)
    ap.add_argument('--all', action='store_true', help='all models and datasets, in one process')
    a = ap.parse_args()
    if a.all:
        for d, C in cfg.CONDITIONS:
            for mdl in MODELS:
                run(mdl, d, C)
    else:
        if not (a.model and a.dataset and a.C):
            ap.error('give --model, --dataset and --C, or --all')
        run(a.model, a.dataset, a.C)
        if a.model != 'MNL-start':
            cfg.record_cost(a.model, a.dataset, a.C, n_fits=1)
