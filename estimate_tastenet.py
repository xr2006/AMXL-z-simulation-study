"""
estimate_tastenet.py: step 4. Trains the TasteNet-MNL benchmark (Section 4.2.1) on one dataset.

The network takes raw z = (INCOME, TOD, DUR), standardized with the training data, and maps it through one
hidden ReLU layer of h units and a linear output to (beta_TT, beta_POI). These tastes enter an MNL with
the cost coefficient fixed at -1. Training minimizes the NLL of the C history choices of the training
drivers, using Adam with weight decay. Early stopping and model selection use the NLL of the evaluation
choices (t = 21..25) of the development drivers, predicted from z alone.

Grid (Section 4.2.2): h in {50, 100, 200} x weight decay in {1e-4, 1e-3, 1e-2} x 10 random
initializations = 90 trials. Selection has two steps. First the (h, weight decay) cell with the lowest
mean development NLL is chosen, so a lucky initialization cannot pick the cell. Then, inside that cell,
the run whose development NLL is closest to the cell mean is kept. Each run is seeded by its restart
number.

Output: results/estimates/tastenet/{dataset}_C{C}_grid.csv  (one row per trial)
        results/estimates/tastenet/{dataset}_C{C}_model.pt  (the selected network: state_dict, h, wd)

Usage: python estimate_tastenet.py --dataset noise20 --C 5 [--num_runs 10]
"""
import argparse
import itertools

import pandas as pd
import torch
from torch.utils.data import DataLoader

import config as cfg
from tastenet.data import ChoiceDataset, load_split
from tastenet.models import TasteNetChoice
from tastenet.train import train, evaluate_epoch

HIDDEN = [50, 100, 200]
WEIGHT_DECAY = [1e-4, 1e-3, 1e-2]
OUT_DIR = cfg.EST_DIR / 'tastenet'


def build(hidden):
    return TasteNetChoice([3, hidden, 2], activation='relu')


def load_model(dataset, C):
    """The selected TasteNet of one dataset (used by evaluate.py)."""
    ck = torch.load(OUT_DIR / f'{cfg.tag(dataset, C)}_model.pt', weights_only=False)
    model = build(ck['hidden'])
    model.load_state_dict(ck['state_dict'])
    return model.eval()


def main(dataset, C, num_runs, batch_size, num_epochs, lr):
    torch.set_num_threads(1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = cfg.tag(dataset, C)
    ds_train = ChoiceDataset(load_split(cfg.DATA_DIR / f'train_{dataset}.csv', t_max=C))
    ds_dev = ChoiceDataset(load_split(cfg.DATA_DIR / f'dev_{dataset}.csv', t_min=cfg.T_HIST + 1))
    assert (ds_train.N, ds_dev.N) == (cfg.SPLIT['train'] * C, cfg.SPLIT['dev'] * cfg.H)
    data_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    eval_train = DataLoader(ds_train, batch_size=batch_size)
    data_dev = DataLoader(ds_dev, batch_size=batch_size)
    z_mean, z_std = ds_train.z.mean(0), ds_train.z.std(0)

    evals, models = [], []
    for hidden, wd, run in itertools.product(HIDDEN, WEIGHT_DECAY, range(num_runs)):
        model = None
        for attempt in range(10):              # retry runs abandoned for a bad initialization
            torch.manual_seed(100 * run + attempt)
            cand = build(hidden)
            cand.taste.z_mean.copy_(z_mean)
            cand.taste.z_std.copy_(z_std)
            model, ok = train(cand, data_train, data_dev, lr, wd, num_epochs)
            if ok:
                break
        if model is None:
            raise RuntimeError(f'hidden={hidden}, wd={wd}, run {run}: no successful run in 10 attempts')
        evals.append({'hidden': hidden, 'wd': wd, 'run': run, 'nll_train': evaluate_epoch(model, eval_train),
                      'nll_dev': evaluate_epoch(model, data_dev)})
        models.append(model)
        print(f"[{tag}] hidden={hidden:3d} wd={wd:g} run={run}  NLL train={evals[-1]['nll_train']:.4f} "
              f"dev={evals[-1]['nll_dev']:.4f}", flush=True)

    grid = pd.DataFrame(evals)
    cell_mean = grid.groupby(['hidden', 'wd'], sort=False)['nll_dev'].mean()
    best_cell = cell_mean.idxmin()
    in_cell = grid[(grid.hidden == best_cell[0]) & (grid.wd == best_cell[1])]
    best = (in_cell['nll_dev'] - cell_mean[best_cell]).abs().idxmin()
    grid['selected'] = grid.index == best
    grid.to_csv(OUT_DIR / f'{tag}_grid.csv', index=False)
    torch.save({'hidden': int(grid.loc[best, 'hidden']), 'wd': float(grid.loc[best, 'wd']),
                'run': int(grid.loc[best, 'run']), 'state_dict': models[best].state_dict()},
               OUT_DIR / f'{tag}_model.pt')
    print(f"[{tag}] selected hidden={best_cell[0]} wd={best_cell[1]:g} run={grid.loc[best, 'run']}  "
          f"dev NLL={grid.loc[best, 'nll_dev']:.4f} (cell mean {cell_mean[best_cell]:.4f})")
    return len(grid)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='TasteNet-MNL on one dataset')
    ap.add_argument('--dataset', choices=list(cfg.NOISE), required=True)
    ap.add_argument('--C', type=int, choices=cfg.C_LIST, required=True)
    ap.add_argument('--num_runs', type=int, default=10, help='random initializations per (h, weight decay) cell')
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--num_epochs', type=int, default=200)
    ap.add_argument('--lr', type=float, default=0.0002)
    a = ap.parse_args()
    n_fits = main(a.dataset, a.C, a.num_runs, a.batch_size, a.num_epochs, a.lr)
    cfg.record_cost('TasteNet', a.dataset, a.C, n_fits=n_fits)
