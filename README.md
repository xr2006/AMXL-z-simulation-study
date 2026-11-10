# AMXL-z simulation study: EV charging station choice

This repository holds the code for the simulation study (Section 4) of

> Ren, X., Clark, C., Wang, Y., Yabe, T., Chow, J.Y.J. *Modeling joint public EV charging location and nearby activity choices using large-scale GPS data.*

The study generates synthetic panels of charging-station choices in which tastes vary nonlinearly with driver characteristics *z* and carry a skewed residual. It then compares **AMXL-z** (agent-specific mixed logit with a *z*-dependent prior, in linear and quadratic forms) with **MNL**, **MXL** and **TasteNet-MNL** on three things: predictive performance, recovery of individual tastes, and computational cost.

One command runs everything, from data generation to the tables and figures of Section 4. It then checks the numbers against the values printed in the paper:

```bash
python run_all.py --jobs 4
```

## What is reproduced

| Paper | Content | Script | Output |
|---|---|---|---|
| Section 4.1, Appendix A.1 | Synthetic datasets (σ ∈ {0.2, 0.5}, 2,000 drivers) | `generate_data.py` | `data/` |
| Figure 2 | Distributions of the true tastes β and the random component η | `dgp_figures.py` | `results/figures/fig2_true_taste_distributions.pdf` |
| Figure A.1 | Systematic tastes against income, by time of day and duration | `dgp_figures.py` | `results/figures/figA1_systematic_tastes.pdf` |
| Figure A.2 | True tastes of all drivers against income | `dgp_figures.py` | `results/figures/figA2_true_tastes_vs_income.pdf` |
| Table A.2 | True tastes by data split | `dgp_figures.py` | `results/tables/tableA2_true_tastes.{tex,csv}` |
| Table 2 | Prediction accuracy (NLL, ACC, ρ²) | `evaluate.py` | `results/tables/table2_prediction.{tex,csv}` |
| Table 3 | Parameter recovery (RMSE of β_TT, β_POI) | `evaluate.py` | `results/tables/table3_recovery.{tex,csv}` |
| Table 4 | CPU time and peak memory of estimation | `evaluate.py` (from `results/compute_cost/`) | `results/tables/table4_compute_cost.{tex,csv}` |
| Figure 3 | Distributions of predicted tastes, out-of-sample (panel) | `evaluate.py` | `results/figures/fig3_predicted_taste_distributions.pdf` |
| — | Reproduced vs. paper values | `check_against_paper.py` | `results/paper_comparison.md` |

## Repository structure

```
.
├── run_all.py               Runs the whole pipeline (steps 1-7 below), in parallel, skipping finished steps
├── config.py                Shared settings: paths, design (σ, C, splits, seed), model names, cost recorder
├── data_utils.py            Shared helpers: load a split, history / evaluation occasions, arrays, basis φ(z)
│
├── generate_data.py         1. Synthetic panel data (Eqs. 13-15, A.1)
├── estimate_mnl_mxl.py      2. MNL and MXL with linear z interactions (xlogit); MNL without z = AMXL start
├── estimate_amxl.py         3. AMXL-linear and AMXL-quadratic: decomposition algorithm, λ search,
│                               and prediction from the empirical distribution of the agents' deviations
├── estimate_tastenet.py     4. TasteNet-MNL: 3 hidden sizes x 3 weight decays x 10 initializations
├── tastenet/                   TasteNet network (models.py), data loader (data.py), training loop (train.py)
├── evaluate.py              5. Scores all models in the three settings -> Tables 2, 3, 4 and Figure 3
├── dgp_figures.py           6. Figure 2, Figures A.1-A.2 and Table A.2 (needs only the data)
├── check_against_paper.py   7. Compares the results with paper_reference/paper_values.csv
│
├── paper_reference/
│   └── paper_values.csv     Values of Tables 2, 3, 4 and A.2 as printed in the paper
├── results/                 Outputs (tables, figures, metrics.csv, paper_comparison.md)
├── requirements.txt         Exact package versions used for the results
└── environment.yml          Conda environment (Python 3.11 + requirements.txt)
```

The pipeline also creates the folders below. They are regenerated and git-ignored.

| Folder | Content |
|---|---|
| `data/` | `{train,dev,test}_{noise20,noise50}.csv` (one row per choice occasion) and `*_truebeta.csv` (true tastes per driver) |
| `results/estimates/` | One folder per model: `mnl_start/`, `mnl/`, `mxl/`, `amxl_linear/`, `amxl_quadratic/`, `tastenet/` |
| `results/compute_cost/` | One JSON per estimation run: CPU time, peak memory, number of fits |
| `results/logs/` | The console output of each step |

## Installation

Python 3.11 was used for the reported results.

```bash
conda env create -f environment.yml
```

```bash
conda activate evcs-sim
```

With pip only:

```bash
pip install -r requirements.txt
```

The AMXL individual step is solved with cvxpy and the Clarabel solver (installed with cvxpy). The cost recorder uses the Unix `resource` module. On Windows, everything runs, but Table 4 is not produced.

## Running

### Everything

```bash
python run_all.py --jobs 4
```

`--jobs` sets how many estimations run in parallel. Each estimation is a separate single-threaded process, so that its CPU time and peak memory can be recorded. The total CPU time is about one hour: TasteNet about 40 min, AMXL about 20 min, MNL/MXL about 1 min. With `--jobs 4` the whole run takes about 20-25 minutes. If a run is interrupted, start it again: finished estimations are skipped. Use `--force` to rerun them all.

### Step by step

Each estimation script handles one dataset, so the steps can also be run, or spread over machines, by hand. `{noise20,noise50}` is σ = 0.2 / 0.5, and `--C` is the number of choices per driver used for estimation.

```bash
python generate_data.py
python estimate_mnl_mxl.py --model MNL-start --dataset noise20 --C 5     # starting value for AMXL
python estimate_mnl_mxl.py --model MNL       --dataset noise20 --C 5
python estimate_mnl_mxl.py --model MXL       --dataset noise20 --C 5
python estimate_amxl.py    --form linear     --dataset noise20 --C 5
python estimate_amxl.py    --form quadratic  --dataset noise20 --C 5
python estimate_tastenet.py                  --dataset noise20 --C 5
# ... repeat for (noise20, 10), (noise50, 5), (noise50, 10)
python evaluate.py
python dgp_figures.py
python check_against_paper.py
```

## Study design in brief

**Data (Section 4.1, Appendix A.1).** 2,000 drivers each make 25 choices among J = 3 unlabeled stations, with V = β_TT·TT + β_POI·POI − CO. Each taste is β_k,n = f_k(z_n) + η_k,n. Here f_k is a polynomial in z = (income, night charging, duration) with pairwise interactions, and η_k,n = σ·s_k(z_n)·u_k,n is heteroskedastic and skewed (u = ±(1 − Exponential(1))). The noise level is σ ∈ {0.2, 0.5}. Drivers are split 1,200 / 400 / 400 into training, development and test sets. Both noise levels share the same seed, so only σ differs.

**Panel structure.** Occasions t = 1-20 form each driver's history pool. A dataset with C ∈ {5, 10} uses t ≤ C for estimation and as the panel for prediction. Occasions t = 21-25 are the five evaluation choices. They are never used for estimation and are the same for every C.

**Models (Section 4.2.1).** All models use the utility of the data-generating process (cost coefficient fixed at −1, no ASCs).

| Paper name | Taste specification | Hyperparameters (selected on the development NLL) |
|---|---|---|
| MNL | β(z) = γ′(1, inc, tod, dur) | none |
| MXL | MNL + N(0, ω²) per driver; 600 Halton draws | none |
| TasteNet | β = NN(z), one hidden ReLU layer | h ∈ {50, 100, 200} × weight decay ∈ {1e-4, 1e-3, 1e-2} × 10 initializations |
| AMXL-linear | agent tastes θ_n shrunk to m(z) = Γ′φ_L(z) | λ ∈ {1e-5, …, 1e5} (11 values) |
| AMXL-quadratic | same with the full quadratic φ_Q(z) | λ ∈ {1e-5, …, 1e5} (11 values) |

**Prediction settings (Section 4.2.2).** *In-sample*: the training drivers. *Out-of-sample (no panel)*: the test drivers, from z alone. *Out-of-sample (panel)*: the test drivers, from z and their C history choices. MXL and AMXL weight their taste distribution by the likelihood of the history. TasteNet and MNL cannot use the history.

**Metrics (Section 4.2.3).** NLL per choice divided by J, accuracy, McFadden's ρ², and the RMSE of the predicted against the true individual tastes. Computational cost: CPU time (user + system) and peak resident memory of each estimation process, including the hyperparameter search.

## Reproduction status

This was checked with a clean run of `python run_all.py --jobs 4`, starting from empty `data/` and `results/` folders. It took 17.8 min on an Apple M2 Pro (macOS, Python 3.11.13, the package versions in `requirements.txt`). The full report is in [`results/paper_comparison.md`](results/paper_comparison.md).

| Paper element | Result |
|---|---|
| Synthetic data | Byte-identical to the data used for the paper |
| Table 2 (216 values) | All identical to the paper |
| Table 3 (120 values) | All identical to the paper |
| Table A.2 (64 values) | All identical to the paper |
| Figures 2, 3, A.1, A.2 | Same as in the paper |
| Table 4 | Machine dependent; the same pattern as the paper (below) |

Table 4 measures each estimation as one process, so its numbers depend on the hardware and on which libraries the process loads:

| Model | Fits | CPU s, σ=0.2 C=5 / C=10 (paper) | CPU s, σ=0.2 C=5 / C=10 (this run) | Peak MB C=5 / C=10 (paper) | Peak MB C=5 / C=10 (this run) |
|---|---:|---|---|---|---|
| MNL | 1 | 3.0 / 2.9 | 1.1 / 1.2 | 226 / 233 | 165 / 165 |
| MXL | 1 | 5.6 / 8.7 | 3.7 / 6.8 | 1,099 / 1,833 | 1,035 / 1,587 |
| TasteNet | 90 | 525 / 869 | 536 / 911 | 465 / 462 | 338 / 338 |
| AMXL-linear | 11 | 133 / 143 | 126 / 137 | 367 / 367 | 307 / 323 |
| AMXL-quadratic | 11 | 200 / 173 | 202 / 178 | 367 / 372 | 309 / 271 |

The paper's conclusions hold. MXL's peak memory grows strongly with C, while AMXL's stays roughly flat. Both AMXL variants need far less CPU time than the TasteNet grid search. The baselines are lower here because the paper's MNL/MXL timings ran in a process that also loaded the AMXL code (cvxpy), whereas here each estimation loads only what it needs. Peak memory also varies by a few tens of MB from run to run.

## Notes on implementation details

- **Evaluation choices.** The paper says the "subsequent five" choices are reserved for evaluation. In the code they are occasions 21-25 for every C, so the evaluation data are identical across C. The occasions are i.i.d., so this is statistically the same as using t = C+1, …, C+5.
- **λ grid.** {10⁻⁵, 10⁻⁴, …, 10⁵} has 11 values. This matches "Fits = 11" in Table 4.
- **Quadratic basis.** TOD is a dummy, so TOD² = TOD. φ_Q therefore has 8 non-constant terms plus the intercept.
- **AMXL starting value.** The population-level tastes start at the estimates of an MNL without z (`MNL-start`), and the interaction terms start at zero.
- **Randomness.** Data generation, the xlogit Halton draws, the MXL prediction draws and the TasteNet initializations are all seeded. With the package versions in `requirements.txt`, the results are deterministic on a given machine. Other platforms or library versions may change the last digits.

## Citation

If you use this code, please cite the paper above.
