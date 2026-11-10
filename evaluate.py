"""
evaluate.py: step 5. Scores every model in the three prediction settings of Section 4.2.2 and writes
Tables 2-4 and Figure 3 of the paper.

Every setting predicts the five evaluation choices (t = 21..25) of each driver. The history is the
driver's first C choices.
    In-sample                   the 1,200 training drivers, whose history was used for estimation
    Out-of-sample (no panel)    the 400 test drivers, predicted from z alone
    Out-of-sample (panel)       the 400 test drivers, predicted from z and their C history choices

How each model predicts:
    MNL             beta(z) = gamma' phi_L(z) in every setting
    MXL             probabilities averaged over 2,000 draws of N(gamma' phi_L(z), omega^2). In the
                    in-sample and panel settings the draws are weighted by the likelihood of the
                    driver's history (the conditional distribution, Revelt and Train, 2000), and the
                    taste is the conditional mean.
    TasteNet        beta = f(z) in every setting; it cannot use a driver's history
    AMXL-*          the empirical distribution {m(z) + r_j} (estimate_amxl.predict_known / _new / _warm)
    True tastes     each driver's true beta (the ceiling)

Metrics (Section 4.2.3): NLL per choice divided by J (Eq. 16), ACC (Eq. 17), McFadden's rho^2 (Eq. 18),
and RMSE between the predicted and the true individual tastes (Eq. 19).

Output:
    results/metrics.csv                                      every model x dataset x setting
    results/tables/table2_prediction.{tex,csv}               Table 2
    results/tables/table3_recovery.{tex,csv}                 Table 3
    results/tables/table4_compute_cost.{tex,csv}             Table 4 (from results/compute_cost/)
    results/figures/fig3_predicted_taste_distributions.pdf   Figure 3

Usage: python evaluate.py
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde

import config as cfg
import data_utils as du
import estimate_amxl as amxl
import estimate_tastenet as tn

TASTES = cfg.TASTES
R_DRAWS = 2000                       # MXL simulation draws for prediction
SIGMA = cfg.NOISE


# ============================================================ predictors
# Each returns (choice probabilities of the evaluation rows (n, J), predicted tastes (drivers, 2)).
def point_probs(b, fut):
    X, _, A, drv = du.arrays(fut)
    return du.mixture_probs(du.with_cost(b)[drv][:, None, :], X, A)


def z_mean(dataset, C, fut, model):
    """gamma' phi_L(z) (drivers, 2) of the drivers of fut, from the MNL or MXL estimate, and its row."""
    r = pd.read_csv(cfg.EST_DIR / model / f'{model}_{cfg.tag(dataset, C)}.csv').iloc[0]
    gamma = np.array([[r[f'gamma_{t}_{k}'] for k in TASTES] for t in du.TERMS['linear']])
    F_tr = du.driver_z(du.history(du.load_split('train', dataset), C))
    return du.basis(du.driver_z(fut), F_tr, 'linear') @ gamma, r


def predict_mnl(dataset, C, s, hist, fut, truth):
    b, _ = z_mean(dataset, C, fut, 'mnl')
    return point_probs(b, fut), b


def predict_mxl(dataset, C, s, hist, fut, truth):
    mean, r = z_mean(dataset, C, fut, 'mxl')
    sd = r[[f'sd_{t}' for t in TASTES]].values.astype(float)
    E = np.random.default_rng(0).standard_normal((R_DRAWS, 2))
    B = du.with_cost(mean[:, None, :] + sd * E[None])            # (drivers, R, K)
    loglik = np.zeros(B.shape[:2])
    if s != 2:                                                   # weight the draws by the history likelihood
        Xh, yh, _, dh = du.arrays(hist)
        for ch in np.array_split(np.arange(len(yh)), max(1, len(yh) // 2000)):
            V = np.einsum('nrk,nkj->nrj', B[dh[ch]], Xh[ch])
            V = V - V.max(-1, keepdims=True)
            np.add.at(loglik, dh[ch], V[np.arange(len(ch)), :, yh[ch]] - np.log(np.exp(V).sum(-1)))
    w = np.exp(loglik - loglik.max(1, keepdims=True))
    w /= w.sum(1, keepdims=True)
    X, _, A, drv = du.arrays(fut)
    P = np.empty((len(drv), X.shape[2]))
    for ch in np.array_split(np.arange(len(drv)), max(1, len(drv) // 2000)):
        Pr = du.softmax_avail(np.einsum('nrk,nkj->nrj', B[drv[ch]], X[ch]), A[ch][:, None, :])
        P[ch] = np.einsum('nr,nrj->nj', w[drv[ch]], Pr)
    return P, (mean if s == 2 else np.einsum('nr,nrk->nk', w, B[:, :, :2]))


def amxl_predictor(form):
    def predict(dataset, C, s, hist, fut, truth):
        if s == 1:
            return amxl.predict_known(dataset, C, hist, fut, form)
        if s == 2:
            return amxl.predict_new(dataset, C, fut, form)
        return amxl.predict_warm(dataset, C, hist, fut, form)
    return predict


def predict_tastenet(dataset, C, s, hist, fut, truth):
    model = tn.load_model(dataset, C)
    with torch.no_grad():
        b = model.taste(torch.tensor(du.driver_z(fut), dtype=torch.float32)).numpy().astype(float)
    return point_probs(b, fut), b


def predict_truth(dataset, C, s, hist, fut, truth):
    b = truth[[f'beta_{t}' for t in TASTES]].values
    return point_probs(b, fut), b


PREDICT = {'MNL': predict_mnl, 'MXL': predict_mxl, 'TasteNet': predict_tastenet,
           'AMXL-linear': amxl_predictor('linear'), 'AMXL-quadratic': amxl_predictor('quadratic'),
           cfg.TRUTH: predict_truth}


def metrics(P, y, b_hat, truth):
    n, J = P.shape
    ll = np.log(np.maximum(P[np.arange(n), y], 1e-300))
    rmse = np.sqrt(((b_hat - truth[[f'beta_{t}' for t in TASTES]].values) ** 2).mean(0))
    return {'NLL': -ll.sum() / n / J, 'ACC': (P.argmax(1) == y).mean(), 'rho2': 1 - ll.sum() / (n * np.log(1 / J)),
            'RMSE_TT': rmse[0], 'RMSE_POI': rmse[1]}


# ============================================================ tables
HEAD = {'NLL': 'NLL', 'ACC': 'ACC', 'rho2': r'$\rho^2$', 'RMSE_TT': r'$\beta_{TT}$', 'RMSE_POI': r'$\beta_{POI}$'}
BEST = {'NLL': min, 'ACC': max, 'rho2': max, 'RMSE_TT': min, 'RMSE_POI': min}


def wide_table(df, cols, with_truth):
    """One row per (sigma, C, model), one column per (setting, metric). The values are not rounded here:
    the tables print them with '%.3f', which rounds the stored binary value as the paper does
    (0.7835 is stored as 0.78349999... and prints as 0.783)."""
    models = cfg.MODELS + ([cfg.TRUTH] if with_truth else [])
    d = df[df.model.isin(models)].copy()
    d['sigma'] = d['dataset'].map(SIGMA)
    t = d.pivot_table(index=['sigma', 'C', 'model'], columns='setting', values=cols, sort=False)
    t = t.reorder_levels([1, 0], axis=1)[[(s, c) for s in cfg.SETTINGS.values() for c in cols]]
    order = [(SIGMA[ds], C, m) for ds, C in cfg.CONDITIONS for m in models]
    return t.loc[order]


def latex_table(t, cols, caption, label):
    settings, nc = list(cfg.SETTINGS.values()), len(cols)
    ncol = len(settings) * nc
    L = [r'\begin{table}[p]', r'\centering', r'\footnotesize', rf'\caption{{{caption}}}', rf'\label{{{label}}}',
         r'\begin{adjustbox}{max width=\linewidth}', r'\begin{tabular}{l' + 'c' * ncol + '}', r'\toprule',
         ' & ' + ' & '.join(rf'\multicolumn{{{nc}}}{{c}}{{{s}}}' for s in settings) + r' \\',
         ''.join(rf'\cmidrule(lr){{{2 + i * nc}-{1 + (i + 1) * nc}}}' for i in range(len(settings))),
         'Model & ' + ' & '.join(HEAD[c] for _ in settings for c in cols) + r' \\', r'\midrule']
    for i, (sigma, C) in enumerate(dict.fromkeys((s, c) for s, c, _ in t.index)):
        if i:
            L.append(r'\addlinespace')
        L.append(rf'\multicolumn{{{ncol + 1}}}{{l}}{{\textit{{$\sigma = {sigma}$, $C = {C}$}}}} \\')
        block = t.loc[(sigma, C)]
        best = {k: f'{BEST[k[1]](block.loc[cfg.MODELS, k]):.3f}' for k in block.columns}   # ties: all bold
        for model, row in block.iterrows():
            cells = []
            for k, v in row.items():
                txt = f'{v:.3f}'
                if model == cfg.TRUTH:
                    txt = rf'\textit{{{txt}}}'
                elif txt == best[k]:
                    txt = rf'\textbf{{{txt}}}'
                cells.append(txt)
            name = rf'\textit{{{model}}}' if model == cfg.TRUTH else model
            L.append(f'{name} & ' + ' & '.join(cells) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{adjustbox}', r'\end{table}']
    return '% Auto-generated by evaluate.py (needs \\usepackage{booktabs} and \\usepackage{adjustbox})\n' + '\n'.join(L) + '\n'


def cost_table():
    """Table 4 from the JSON records of results/compute_cost/ (None if some are missing)."""
    files = sorted(cfg.COST_DIR.glob('*.json'))
    if not files:
        return None
    cost = pd.DataFrame([json.load(open(f)) for f in files])
    cost['sigma'] = cost['dataset'].map(SIGMA)
    if len(cost[cost.model.isin(cfg.MODELS)]) < len(cfg.MODELS) * len(cfg.CONDITIONS):
        print('compute cost: some runs are not recorded, Table 4 is incomplete')
    rows = []
    for m in cfg.MODELS:
        c = cost[cost.model == m]
        if c.empty:
            continue
        row = {'Model': m, 'Fits': int(c['n_fits'].iloc[0])}
        for sigma in SIGMA.values():
            for C in cfg.C_LIST:
                v = c[(c.sigma == sigma) & (c.C == C)]['cpu_s']
                row[f'CPU s, sigma={sigma}, C={C}'] = v.iloc[0] if len(v) else np.nan
        for C in cfg.C_LIST:
            row[f'Peak MB, C={C}'] = c[c.C == C]['peak_MB'].max()      # the larger of the two noise levels
        rows.append(row)
    t = pd.DataFrame(rows).set_index('Model')
    fmt_t = lambda v: '--' if np.isnan(v) else (f'{v:.1f}' if v < 100 else f'{v:,.0f}')
    Cs = cfg.C_LIST
    machine = ', '.join(sorted(cost['machine'].unique()))
    L = [r'\begin{table}[t]', r'\centering', r'\footnotesize', r'\setlength{\tabcolsep}{4pt}',
         rf'\caption{{Computation time and peak RAM usage of model training/estimation (1,200 training drivers). '
         rf'Measured on {machine}, one thread per process.}}', r'\label{tab:cost}',
         r'\begin{tabular}{lr' + 'r' * 3 * len(Cs) + '}', r'\toprule',
         r' & & ' + ' & '.join([rf'\multicolumn{{{len(Cs)}}}{{c}}{{CPU time (s), $\sigma = {s}$}}' for s in SIGMA.values()]
                               + [rf'\multicolumn{{{len(Cs)}}}{{c}}{{Peak memory (MB)}}']) + r' \\',
         ''.join(rf'\cmidrule(lr){{{3 + i * len(Cs)}-{2 + (i + 1) * len(Cs)}}}' for i in range(3)),
         'Model & Fits & ' + ' & '.join(f'$C = {C}$' for _ in range(3) for C in Cs) + r' \\', r'\midrule']
    for m, r in t.iterrows():
        vals = [fmt_t(r[f'CPU s, sigma={s}, C={C}']) for s in SIGMA.values() for C in Cs] + \
               [f"{r[f'Peak MB, C={C}']:,.0f}" for C in Cs]
        L.append(f"{m} & {r['Fits']} & " + ' & '.join(vals) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return t, '% Auto-generated by evaluate.py (needs \\usepackage{booktabs})\n' + '\n'.join(L) + '\n'


# ============================================================ Figure 3
COLORS = {'MXL': dict(color='#1baf7a'), 'TasteNet': dict(color='#2a78d6'), 'AMXL-linear': dict(color='#99007a'),
          'AMXL-quadratic': dict(color='#eb6834'), cfg.TRUTH: dict(color='gray')}
LW, LW_TRUE = 1.2, 1.6
DIST_BW = 0.5            # bandwidth: this fraction of Scott's bandwidth for the panel's true tastes
TASTE_LABEL = {'TT': r'$\beta_{\mathrm{TT}}$', 'POI': r'$\beta_{\mathrm{POI}}$'}
RC = {'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
      'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold',
      'mathtext.sf': 'Arial', 'mathtext.cal': 'Arial', 'mathtext.tt': 'DejaVu Sans Mono',
      'font.size': 11, 'axes.titlesize': 12, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
      'axes.labelsize': 11.5, 'xtick.labelsize': 10.5, 'ytick.labelsize': 10.5, 'legend.fontsize': 10,
      'text.color': '#0b0b0b', 'axes.labelcolor': '#0b0b0b', 'axes.edgecolor': '#333333',
      'xtick.color': '#222222', 'ytick.color': '#222222', 'axes.linewidth': 0.9, 'axes.spines.top': False,
      'axes.spines.right': False, 'axes.grid': False, 'xtick.major.size': 4, 'xtick.major.width': 0.9,
      'ytick.major.size': 4, 'ytick.major.width': 0.9, 'legend.frameon': False,
      'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02, 'pdf.fonttype': 42, 'ps.fonttype': 42}


def kde(values, grid, h):
    return gaussian_kde(values, bw_method=h / values.std(ddof=1))(grid)


def figure3(tastes, truths, s, path):
    """Kernel densities of the tastes predicted for the test drivers in setting s by MXL, TasteNet,
    AMXL-linear and AMXL-quadratic, against the true tastes. Rows: beta_TT, beta_POI; columns: datasets."""
    with plt.rc_context(RC):
        fig, axes = plt.subplots(2, len(cfg.CONDITIONS), figsize=(13.3, 5.8), layout='constrained', squeeze=False)
        for r, k in enumerate(TASTES):
            j = TASTES.index(k)
            pooled = np.concatenate([truths[(d, s)][f'beta_{k}'].values for d in SIGMA])
            lo, hi = np.quantile(pooled, [0.002, 0.998])
            grid = np.linspace(lo - 0.12 * (hi - lo), hi + 0.12 * (hi - lo), 400)
            for c, (d, C) in enumerate(cfg.CONDITIONS):
                ax = axes[r, c]
                true = truths[(d, s)][f'beta_{k}'].values
                h = DIST_BW * true.std(ddof=1) * len(true) ** -0.2
                top = 0.0
                for m, st in COLORS.items():                  # the true tastes last, under the models
                    is_true = m == cfg.TRUTH
                    dens = kde(tastes[(d, C, s)][m][:, j], grid, h)
                    ax.plot(grid, dens, color=st['color'], ls='-', lw=LW_TRUE if is_true else LW,
                            zorder=2 if is_true else 3, label=m)
                    top = max(top, dens.max())
                ax.set_xlim(grid[0], grid[-1])
                ax.set_ylim(0, 1.08 * top)
                ax.yaxis.set_major_locator(MaxNLocator(4))
                ax.set_title(rf'$\sigma$ = {SIGMA[d]}, C = {C}:  {TASTE_LABEL[k]}')
                if c == 0:
                    ax.set_ylabel('Density')
                if r == 1:
                    ax.set_xlabel('taste parameter')
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='outside lower center', ncol=len(COLORS),
                   frameon=False, handlelength=2.6, columnspacing=2.2)
        fig.suptitle(f'Predicted tastes: {cfg.SETTINGS[s]}', fontsize=13, fontweight='normal')
        fig.savefig(path)
    plt.close(fig)


# ============================================================ main
def main():
    rows, tastes, truths = [], {}, {}
    for dataset in cfg.NOISE:
        data = {sp: (du.load_split(sp, dataset), du.load_truth(sp, dataset)) for sp in ['train', 'test']}
        for C in cfg.C_LIST:
            for s, setting in cfg.SETTINGS.items():
                w, truth = data['train'] if s == 1 else data['test']
                hist, fut = (None if s == 2 else du.history(w, C)), du.future(w)
                truths[(dataset, s)], tastes[(dataset, C, s)] = truth, {}
                for model, predict in PREDICT.items():
                    P, b = predict(dataset, C, s, hist, fut, truth)
                    tastes[(dataset, C, s)][model] = b
                    rows.append({'dataset': dataset, 'sigma': SIGMA[dataset], 'C': C, 'setting': setting,
                                 'model': model, **metrics(P, fut['CHOICE'].values, b, truth)})
            print(f'scored {cfg.tag(dataset, C)}')
    df = pd.DataFrame(rows)
    cfg.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.RESULTS_DIR / 'metrics.csv', index=False)

    t2 = wide_table(df, ['NLL', 'ACC', 'rho2'], with_truth=True)
    t3 = wide_table(df, ['RMSE_TT', 'RMSE_POI'], with_truth=False)
    t2.to_csv(cfg.TABLE_DIR / 'table2_prediction.csv', float_format='%.3f')
    t3.to_csv(cfg.TABLE_DIR / 'table3_recovery.csv', float_format='%.3f')
    (cfg.TABLE_DIR / 'table2_prediction.tex').write_text(latex_table(
        t2, ['NLL', 'ACC', 'rho2'], 'Prediction accuracy. Bold: best model in each column of a block; italics: the '
        'true tastes (the ceiling).', 'tab:prediction'))
    (cfg.TABLE_DIR / 'table3_recovery.tex').write_text(latex_table(
        t3, ['RMSE_TT', 'RMSE_POI'], 'Parameter recovery: RMSE between the predicted and the true individual '
        'tastes. Bold: best model in each column of a block.', 'tab:recovery'))
    cost = cost_table()
    if cost is not None:
        cost[0].to_csv(cfg.TABLE_DIR / 'table4_compute_cost.csv')
        (cfg.TABLE_DIR / 'table4_compute_cost.tex').write_text(cost[1])
    figure3(tastes, truths, 3, cfg.FIG_DIR / 'fig3_predicted_taste_distributions.pdf')

    with pd.option_context('display.width', 200, 'display.max_columns', 20):
        print('\nTable 2: prediction accuracy\n', t2.to_string(float_format='%.3f'))
        print('\nTable 3: parameter recovery (RMSE)\n', t3.to_string(float_format='%.3f'))
        if cost is not None:
            print('\nTable 4: computation cost\n', cost[0].round(1).to_string())
    print(f'\nsaved -> {cfg.RESULTS_DIR}/metrics.csv, {cfg.TABLE_DIR}/, {cfg.FIG_DIR}/')


if __name__ == '__main__':
    main()
