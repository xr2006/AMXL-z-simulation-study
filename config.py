"""
config.py: settings shared by all scripts. They cover paths, the experimental design of Section 4.2.2,
model names, and the helper that records CPU time and peak memory (Table 4).
"""
import json
import platform
import sys
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data'                            # synthetic datasets (generate_data.py)
RESULTS_DIR = ROOT / 'results'
EST_DIR = RESULTS_DIR / 'estimates'                 # model estimates, one folder per model
COST_DIR = RESULTS_DIR / 'compute_cost'             # one JSON per estimation run (Table 4)
TABLE_DIR = RESULTS_DIR / 'tables'
FIG_DIR = RESULTS_DIR / 'figures'
LOG_DIR = RESULTS_DIR / 'logs'
PAPER_REF = ROOT / 'paper_reference' / 'paper_values.csv'

# ---------------------------------------------------------------- experimental design (Section 4.2.2)
J = 3                                               # charging stations per choice
NOISE = {'noise20': 0.2, 'noise50': 0.5}            # dataset tag -> noise level sigma
C_LIST = [5, 10]                                    # choices per driver used for estimation
CONDITIONS = [(d, C) for d in NOISE for C in C_LIST]
N_DRIVERS = 2000
SPLIT = {'train': 1200, 'dev': 400, 'test': 400}    # 60 : 20 : 20
T_HIST = 20                                         # occasions t = 1..20 are the history pool;
                                                    # a dataset with C choices uses t <= C
H = 5                                               # evaluation choices t = 21..25, the same for every C
T = T_HIST + H                                      # choice occasions generated per driver
SEED = 7

# ---------------------------------------------------------------- models (paper names)
MODELS = ['MNL', 'MXL', 'TasteNet', 'AMXL-linear', 'AMXL-quadratic']
TRUTH = 'True tastes'
SETTINGS = {1: 'In-sample', 2: 'Out-of-sample (no panel)', 3: 'Out-of-sample (panel)'}
TASTES = ['TT', 'POI']


def tag(dataset, C):
    return f'{dataset}_C{C}'


# ---------------------------------------------------------------- computational cost (Table 4)
def cpu_name():
    """CPU model, e.g. 'Apple M2 Pro' (for the caption of Table 4)."""
    try:
        if sys.platform == 'darwin':
            import subprocess
            return subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                                  capture_output=True, text=True).stdout.strip()
        if sys.platform.startswith('linux'):
            for line in open('/proc/cpuinfo'):
                if line.startswith('model name'):
                    return line.split(':', 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def record_cost(model, dataset, C, n_fits):
    """Write the CPU time (user + system) and peak resident memory of the running process to
    results/compute_cost/{model}_{dataset}_C{C}.json. Call it at the end of a script that estimates one
    model on one dataset, so that each number covers one process (imports and data loading included).
    Needs the `resource` module (macOS / Linux); on other systems nothing is written."""
    try:
        import resource
    except ImportError:
        print('record_cost: the resource module is not available, no cost recorded')
        return
    ru = resource.getrusage(resource.RUSAGE_SELF)
    peak = ru.ru_maxrss / (1024 ** 2 if sys.platform == 'darwin' else 1024)   # bytes on macOS, KB on Linux
    COST_DIR.mkdir(parents=True, exist_ok=True)
    row = {'model': model, 'dataset': dataset, 'C': C, 'n_fits': n_fits,
           'cpu_s': round(ru.ru_utime + ru.ru_stime, 2),
           'peak_MB': round(peak, 1), 'machine': f'{cpu_name()} ({platform.system()} {platform.machine()})',
           'python': platform.python_version()}
    with open(COST_DIR / f'{model}_{dataset}_C{C}.json', 'w') as f:
        json.dump(row, f, indent=1)
    print(f"[cost] {model} {dataset} C={C}: CPU {row['cpu_s']:.1f} s, peak memory {row['peak_MB']:.0f} MB")
