"""
dgp_figures.py: step 6. Figures and the table that describe the synthetic data (Section 4.1 and
Appendix A.1). They need only data/, not the model estimates.

Output:
    results/figures/fig2_true_taste_distributions.pdf   Figure 2: densities of the true tastes beta_n and
                                                         of their random component eta_n
    results/figures/figA1_systematic_tastes.pdf          Figure A.1: systematic tastes f_k(z) against income,
                                                         by time of day and charging duration
    results/figures/figA2_true_tastes_vs_income.pdf      Figure A.2: true tastes of all 2,000 drivers
                                                         against income
    results/tables/tableA2_true_tastes.{tex,csv}         Table A.2: mean and SD of beta_n and the 95% interval
                                                         of eta_n, by data split

Usage: python dgp_figures.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde, norm

import config as cfg
import generate_data as gen

TAGS = list(cfg.NOISE)
SIGMA = cfg.NOISE
TASTES = gen.TASTES

INK, INK2, MUTED, GRID, AXIS = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
COLOR = {'noise20': '#2a78d6', 'noise50': '#eb6834'}      # sigma = 0.2 / 0.5
TASTE_COLOR = ['tab:blue', 'tab:orange']
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic',
    'mathtext.bf': 'Arial:bold', 'mathtext.sf': 'Arial', 'mathtext.cal': 'Arial', 'mathtext.tt': 'DejaVu Sans Mono',
    'font.size': 8.5, 'axes.titlesize': 8.5, 'axes.labelsize': 8.5, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'legend.fontsize': 8, 'legend.title_fontsize': 8, 'text.color': INK, 'axes.labelcolor': INK,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.edgecolor': AXIS, 'axes.linewidth': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'axes.grid.axis': 'y',
    'grid.color': GRID, 'grid.linewidth': 0.5, 'axes.axisbelow': True, 'axes.titlelocation': 'left',
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'lines.linewidth': 1.5, 'lines.solid_capstyle': 'round', 'lines.solid_joinstyle': 'round',
    'legend.frameon': False, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
    'pdf.fonttype': 42, 'ps.fonttype': 42})
W = 6.5                                                   # text width (in)
TASTE_NAME = {'TT': 'Travel time', 'POI': 'Points of interest'}
LAB_BETA = {'TT': r'$\beta_{TT}$ (\$/min)', 'POI': r'$\beta_{POI}$ (\$/POI)'}
LAB_ETA = {'TT': r'$\eta_{TT}$ (\$/min)', 'POI': r'$\eta_{POI}$ (\$/POI)'}
SPLITS = {'train': 'Training', 'dev': 'Development', 'test': 'Test'}
DURS = [30, 60, 90]


def load_truths():
    """True tastes per (tag, split), plus 'all' = the 2,000 drivers."""
    out = {}
    for tag in TAGS:
        for sp in SPLITS:
            out[(tag, sp)] = pd.read_csv(cfg.DATA_DIR / f'{sp}_{tag}_truebeta.csv')
        out[(tag, 'all')] = pd.concat([out[(tag, sp)] for sp in SPLITS], ignore_index=True)
    return out


def kde_axis(values, q=0.002):
    lo, hi = np.quantile(np.concatenate(values), [q, 1 - q])
    return np.linspace(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo), 400)


def figure2(ALL, path):
    fig, axes = plt.subplots(2, 2, figsize=(W, 4), layout='constrained')
    for c, k in enumerate(TASTES):
        ax = axes[0, c]                                   # (a, b) true tastes and their systematic part
        grid = kde_axis([ALL[tag][f'beta_{k}'] for tag in TAGS])
        f = ALL[TAGS[0]][f'f_{k}'].values                 # identical at both noise levels
        ax.plot(grid, gaussian_kde(f)(grid), color=MUTED, lw=0.8)
        for tag in TAGS:
            ax.plot(grid, gaussian_kde(ALL[tag][f'beta_{k}'].values)(grid), color=COLOR[tag], lw=1)
        ax.set_title(f"({'ab'[c]}) {TASTE_NAME[k]}: true taste parameters")
        ax.set_xlabel(LAB_BETA[k])
        ax2 = axes[1, c]                                  # (c, d) random component eta = beta - f(z)
        eta = {tag: (ALL[tag][f'beta_{k}'] - ALL[tag][f'f_{k}']).values for tag in TAGS}
        grid2 = kde_axis(list(eta.values()))
        for tag in TAGS:
            ax2.plot(grid2, gaussian_kde(eta[tag])(grid2), color=COLOR[tag], lw=1)
        ax2.set_title(rf"({'cd'[c]}) {TASTE_NAME[k]}: random component, $\eta$")
        ax2.set_xlabel(LAB_ETA[k])
        for a, g in [(ax, grid), (ax2, grid2)]:
            a.axvline(0, color=AXIS, linestyle='--', lw=0.6, zorder=1)
            a.set_xlim(g[0], g[-1])
    for r in range(2):
        axes[r, 0].set_ylabel('Density')
    fig.legend(handles=[Line2D([], [], color=MUTED, lw=0.8, label='Systematic part of taste parameter')]
               + [Line2D([], [], color=COLOR[t], lw=0.8, label=rf'$\sigma$ = {SIGMA[t]}') for t in TAGS],
               loc='outside upper center', ncol=4)
    fig.savefig(path)
    plt.close(fig)


def income_grid():
    z99 = norm.ppf(0.99)                                  # 1st-99th percentiles of income
    return np.linspace(*np.exp(gen.INC_MEAN_LOG + np.array([-z99, z99]) * gen.INC_SD_LOG), 200)


def figureA1(path):
    income = income_grid()
    style = {30: '-', 60: '--', 90: ':'}
    name = {'TT': r'$\beta_{TT}$', 'POI': r'$\beta_{POI}$'}
    fig, axes = plt.subplots(2, 2, figsize=(W, 4.1), sharex=True, sharey='col', layout='constrained')
    for r, tod in enumerate([0, 1]):
        for c, k in enumerate(TASTES):
            ax = axes[r, c]
            for dur, ls in style.items():
                Z = gen.standardize_z(income, np.full(income.size, tod), np.full(income.size, dur))
                ax.plot(income, gen.phi(Z) @ gen.COEF[k], color=TASTE_COLOR[c], ls=ls)
            ax.axhline(0, color=AXIS, lw=0.6, zorder=1)
            ax.set_title(f"({'abcd'[2 * r + c]}) Systematic part of {name[k]}, {'night' if tod else 'daytime'}")
        axes[r, 0].set_ylabel('parameter value')
    for ax in axes[1]:
        ax.set_xlabel(r'Annual income (\$k)')
    handles = [Line2D([], [], color='0.3', ls=ls, label=f'{dur} min') for dur, ls in style.items()]
    fig.legend(handles=handles, title='Charging duration', loc='outside upper center', ncol=3, handlelength=2.8)
    fig.savefig(path)
    plt.close(fig)


def income_profile(k, income):
    """Population mean taste at each income: f_k averaged over TOD and DUR (independent of income)."""
    return np.mean([gen.phi(gen.standardize_z(income, np.full(income.size, tod), np.full(income.size, dur)))
                    @ gen.COEF[k] for tod in [0, 1] for dur in DURS], axis=0)


def figureA2(ALL, path):
    inc = income_grid()
    fig, axes = plt.subplots(2, 2, figsize=(W, 4.3), sharex=True, sharey='col', layout='constrained')
    for r, tag in enumerate(TAGS):
        t = ALL[tag]
        g = t.groupby(pd.qcut(t['INCOME'], 10), observed=True)
        for c, k in enumerate(TASTES):
            ax = axes[r, c]
            ax.scatter(t['INCOME'], t[f'beta_{k}'], s=4, color=TASTE_COLOR[c], alpha=0.3, lw=0, zorder=2)
            ax.plot(g['INCOME'].mean(), g[f'beta_{k}'].mean(), 'o', ms=3.5, color=INK, mec='white', mew=1.0, zorder=4)
            ax.plot(inc, income_profile(k, inc), color=INK, lw=1, zorder=3)
            ax.axhline(0, color=AXIS, lw=0.6, zorder=1)
            ax.set_xlim(inc[0], inc[-1])
            ax.set_title(rf"({'abcd'[2 * r + c]}) {TASTE_NAME[k]}, $\sigma$ = {SIGMA[tag]}")
            ax.set_ylabel(LAB_BETA[k])
    for ax in axes[1]:
        ax.set_xlabel(r'Annual income (\$k)')
    fig.legend(handles=[Line2D([], [], ls='', marker='o', ms=3, color=MUTED, label=r'Individual taste $\beta_i$'),
                        Line2D([], [], ls='', marker='o', ms=4, color=INK, mec='white', label='Income-decile mean'),
                        Line2D([], [], color=INK, lw=1, label='Population mean at each income')],
               loc='outside upper center', ncol=3)
    fig.savefig(path)
    plt.close(fig)


def tableA2(truths):
    rows = []
    for tag in TAGS:
        for sp, label in {**SPLITS, 'all': 'All'}.items():
            t = truths[(tag, sp)]
            row = {'sigma': SIGMA[tag], 'sample': f'{label} ({len(t)})'}
            for k in TASTES:
                b, eta = t[f'beta_{k}'].values, (t[f'beta_{k}'] - t[f'f_{k}']).values
                lo, hi = np.quantile(eta, [0.025, 0.975])
                row.update({f'beta_{k}_mean': b.mean(), f'beta_{k}_sd': b.std(),
                            f'eta_{k}_q025': lo, f'eta_{k}_q975': hi})
            rows.append(row)
    df = pd.DataFrame(rows)
    num = lambda v: f'$-${abs(v):.3f}' if v < 0 else f'{v:.3f}'          # LaTeX minus sign
    L = [r'\begin{table}[t]', r'\centering', r'\footnotesize',
         r'\caption{Summary of true individual-level tastes and the random component across data splits}',
         r'\label{tab:true-tastes}', r'\begin{tabular}{lcccccc}', r'\toprule',
         r' & \multicolumn{3}{c}{$\beta_{TT,n}$ (\$/min)} & \multicolumn{3}{c}{$\beta_{POI,n}$ (\$/POI)} \\',
         r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}',
         r'Sample (drivers) & Mean & SD & 95\% CI of $\eta$ & Mean & SD & 95\% CI of $\eta$ \\', r'\midrule']
    for tag in TAGS:
        L.append(rf'\multicolumn{{7}}{{l}}{{\textit{{$\sigma = {SIGMA[tag]}$}}}} \\')
        for _, r in df[df.sigma == SIGMA[tag]].iterrows():
            cells = []
            for k in TASTES:
                cells += [num(r[f'beta_{k}_mean']), num(r[f'beta_{k}_sd']),
                          f"[{num(r[f'eta_{k}_q025'])}, {num(r[f'eta_{k}_q975'])}]"]
            L.append(f"{r['sample']} & " + ' & '.join(cells) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return df, '% Auto-generated by dgp_figures.py (needs \\usepackage{booktabs})\n' + '\n'.join(L) + '\n'


def main():
    cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
    cfg.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    truths = load_truths()
    ALL = {tag: truths[(tag, 'all')] for tag in TAGS}
    figure2(ALL, cfg.FIG_DIR / 'fig2_true_taste_distributions.pdf')
    figureA1(cfg.FIG_DIR / 'figA1_systematic_tastes.pdf')
    figureA2(ALL, cfg.FIG_DIR / 'figA2_true_tastes_vs_income.pdf')
    df, tex = tableA2(truths)
    df.to_csv(cfg.TABLE_DIR / 'tableA2_true_tastes.csv', index=False, float_format='%.3f')
    (cfg.TABLE_DIR / 'tableA2_true_tastes.tex').write_text(tex)
    with pd.option_context('display.width', 200):
        print('Table A.2\n', df.to_string(index=False, float_format='%.3f'))
    print(f'saved -> {cfg.FIG_DIR}/fig2_*, figA1_*, figA2_*; {cfg.TABLE_DIR}/tableA2_true_tastes.*')


if __name__ == '__main__':
    main()
