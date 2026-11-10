"""
check_against_paper.py: step 7. Compares the reproduced numbers with the values printed in the paper
(paper_reference/paper_values.csv, transcribed from Tables 2, 3, 4 and A.2).

Tables 2, 3 and A.2 are compared after rounding to the paper's three decimals. Table 4 (CPU time and
memory) depends on the machine, so it is listed side by side without a pass/fail check.

Output: results/paper_comparison.md, and a summary on screen.

Usage: python check_against_paper.py [--tol 0.001]
"""
import argparse
import json

import numpy as np
import pandas as pd

import config as cfg


def compare_tables(ref, tol):
    """Rows of Tables 2, 3 and A.2: paper value, reproduced value, absolute difference."""
    met = pd.read_csv(cfg.RESULTS_DIR / 'metrics.csv')
    a2 = pd.read_csv(cfg.TABLE_DIR / 'tableA2_true_tastes.csv')
    a2['split'] = a2['sample'].str.split(' ').str[0].str.lower()
    out = []
    for _, r in ref[ref.table.isin(['Table 2', 'Table 3', 'Table A.2'])].iterrows():
        if r.table == 'Table A.2':
            m = a2[(a2.sigma == float(r.sigma)) & (a2.split == r.model)]
        else:
            m = met[(met.sigma == float(r.sigma)) & (met.C == int(float(r.C))) & (met.model == r.model)
                    & (met.setting == r.setting)]
        v = m[r.metric].iloc[0] if len(m) else np.nan
        shown = float(f'{v:.3f}')                     # printed as in the paper's tables
        out.append({**r.to_dict(), 'reproduced': v, 'abs_diff': abs(shown - r.value)})
    df = pd.DataFrame(out)
    df['ok'] = df['abs_diff'] <= tol + 1e-9
    return df


def cost_rows(ref):
    files = sorted(cfg.COST_DIR.glob('*.json'))
    if not files:
        return None
    cost = pd.DataFrame([json.load(open(f)) for f in files])
    cost['sigma'] = cost['dataset'].map(cfg.NOISE)
    rows = []
    for _, r in ref[(ref.table == 'Table 4') & (ref.metric != 'fits')].iterrows():
        c = cost[(cost.model == r.model) & (cost.C == int(float(r.C)))]
        if r.metric == 'cpu_s':
            c = c[c.sigma == float(r.sigma)]
            v = c['cpu_s'].iloc[0] if len(c) else np.nan
        else:
            v = c['peak_MB'].max() if len(c) else np.nan
        rows.append({'model': r.model, 'metric': r.metric, 'sigma': r.sigma, 'C': int(float(r.C)),
                     'paper': r.value, 'reproduced': v, 'ratio': v / r.value})
    return pd.DataFrame(rows), ', '.join(sorted(cost['machine'].unique()))


def main(tol):
    ref = pd.read_csv(cfg.PAPER_REF)
    df = compare_tables(ref, tol)
    L = ['# Reproduced results vs. the paper', '',
         f'Values are rounded to three decimals, as in the paper. A cell matches if |difference| <= {tol}.', '',
         '| Table | Cells | Identical after rounding | Within tolerance | Max abs. difference |',
         '|---|---:|---:|---:|---:|']
    for t, g in df.groupby('table', sort=False):
        L.append(f"| {t} | {len(g)} | {(g.abs_diff < 1e-9).sum()} | {g.ok.sum()} | {g.abs_diff.max():.3f} |")
    bad = df[~df.ok]
    if len(bad):
        L += ['', 'Cells outside the tolerance:', '', '| Table | sigma | C | Model | Setting | Metric | Paper | Reproduced |',
              '|---|---|---|---|---|---|---:|---:|']
        L += [f'| {r.table} | {r.sigma} | {r.C} | {r.model} | {r.setting} | {r.metric} | {r.value:.3f} | {r.reproduced:.4f} |'
              for _, r in bad.iterrows()]
    cost = cost_rows(ref)
    if cost is not None:
        c, machine = cost
        L += ['', f'## Table 4 (machine dependent; reproduced on {machine})', '',
              '| Model | Quantity | sigma | C | Paper | Reproduced | Ratio |', '|---|---|---|---|---:|---:|---:|']
        L += [f"| {r.model} | {'CPU time (s)' if r.metric == 'cpu_s' else 'Peak memory (MB)'} | "
              f"{r.sigma if r.metric == 'cpu_s' else 'max'} | {r.C} | {r.paper:,.1f} | {r.reproduced:,.1f} | {r.ratio:.2f} |"
              for _, r in c.iterrows()]
    report = '\n'.join(L) + '\n'
    (cfg.RESULTS_DIR / 'paper_comparison.md').write_text(report)
    print(report)
    print(f'saved -> {cfg.RESULTS_DIR}/paper_comparison.md')
    return len(bad) == 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tol', type=float, default=0.001, help='largest accepted |difference| after rounding')
    raise SystemExit(0 if main(ap.parse_args().tol) else 1)
