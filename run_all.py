"""
run_all.py: runs the whole simulation study of Section 4, from data generation to the comparison with
the paper.

    1. generate_data.py                  data/                                  (seconds)
    2. estimate_mnl_mxl.py               MNL-start (AMXL starting value), MNL, MXL   (~1 min in total)
    3. estimate_amxl.py                  AMXL-linear, AMXL-quadratic            (~20 min of CPU in total)
       estimate_tastenet.py              TasteNet, 90 trials per dataset        (~40 min of CPU in total)
    4. evaluate.py                       Tables 2-4, Figure 3                   (~1 min)
       dgp_figures.py                    Figure 2, Figures A.1-A.2, Table A.2   (seconds)
       check_against_paper.py            results/paper_comparison.md

Each estimation runs in its own process with one thread (OMP/MKL/BLAS/torch), so that its CPU time and
peak memory can be recorded for Table 4. Independent estimations run in parallel (--jobs). Finished
estimations are skipped, so an interrupted run can be restarted; --force reruns them all. The output
of every process goes to results/logs/.

Usage: python run_all.py [--jobs 4] [--force]
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import config as cfg

PY = sys.executable
ENV = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
       'VECLIB_MAXIMUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1', 'PYTHONHASHSEED': '0'}


def task(name, args, done):
    """A step: log name, command line, output file that marks it as done."""
    return {'name': name, 'cmd': [PY, *args], 'done': done}


def estimation_tasks():
    est = cfg.EST_DIR
    start, rest = [], []
    for d, C in cfg.CONDITIONS:
        t = cfg.tag(d, C)
        dc = ['--dataset', d, '--C', str(C)]
        start.append(task(f'mnl_start_{t}', ['estimate_mnl_mxl.py', '--model', 'MNL-start', *dc],
                          est / 'mnl_start' / f'mnl_start_{t}.csv'))
        rest += [task(f'tastenet_{t}', ['estimate_tastenet.py', *dc], est / 'tastenet' / f'{t}_model.pt'),
                 task(f'amxl_quadratic_{t}', ['estimate_amxl.py', '--form', 'quadratic', *dc],
                      est / 'amxl_quadratic' / f'{t}_best.csv'),
                 task(f'amxl_linear_{t}', ['estimate_amxl.py', '--form', 'linear', *dc],
                      est / 'amxl_linear' / f'{t}_best.csv'),
                 task(f'mxl_{t}', ['estimate_mnl_mxl.py', '--model', 'MXL', *dc], est / 'mxl' / f'mxl_{t}.csv'),
                 task(f'mnl_{t}', ['estimate_mnl_mxl.py', '--model', 'MNL', *dc], est / 'mnl' / f'mnl_{t}.csv')]
    order = ['tastenet', 'amxl_quadratic', 'amxl_linear', 'mxl', 'mnl']      # longest first
    rest.sort(key=lambda x: next(i for i, p in enumerate(order) if x['name'].startswith(p)))
    return start, rest


def execute(t):
    cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(cfg.LOG_DIR / f"{t['name']}.log", 'w') as log:
        log.write('$ ' + ' '.join(t['cmd'][1:]) + '\n')
        log.flush()
        rc = subprocess.run(t['cmd'], cwd=cfg.ROOT, env=ENV, stdout=log, stderr=subprocess.STDOUT).returncode
    return t, rc, time.time() - t0


def run_pool(tasks, jobs, force):
    todo = [t for t in tasks if force or not t['done'].exists()]
    for t in tasks:
        if t not in todo:
            print(f"  skip {t['name']} (done)")
    failed = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for fut in as_completed([pool.submit(execute, t) for t in todo]):
            t, rc, sec = fut.result()
            print(f"  {'done' if rc == 0 else 'FAILED'} {t['name']} ({sec:.0f} s)", flush=True)
            if rc:
                failed.append(t['name'])
    if failed:
        sys.exit(f"failed: {', '.join(failed)}; see {cfg.LOG_DIR}/<name>.log")


def main(jobs, force):
    t0 = time.time()
    print('1. data')
    data_ok = all((cfg.DATA_DIR / f'{s}_{d}.csv').exists() for s in cfg.SPLIT for d in cfg.NOISE)
    run_pool([task('generate_data', ['generate_data.py'], cfg.DATA_DIR / 'test_noise50_truebeta.csv')],
             1, force or not data_ok)
    start, rest = estimation_tasks()
    print(f'2. MNL starting values for AMXL ({jobs} parallel jobs)')
    run_pool(start, jobs, force)
    print(f'3. estimation: TasteNet, AMXL, MXL, MNL ({jobs} parallel jobs)')
    run_pool(rest, jobs, force)
    print('4. evaluation')
    for script in ['evaluate.py', 'dgp_figures.py', 'check_against_paper.py']:
        t, rc, sec = execute(task(script.replace('.py', ''), [script], None))
        print(f"  {'done' if rc == 0 else 'FAILED'} {script} ({sec:.0f} s)")
        if rc and script != 'check_against_paper.py':
            sys.exit(f"{script} failed; see {cfg.LOG_DIR}/{t['name']}.log")
    print(f'\nfinished in {(time.time() - t0) / 60:.1f} min. Comparison with the paper: '
          f'{cfg.RESULTS_DIR / "paper_comparison.md"}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Run the full simulation study')
    ap.add_argument('--jobs', type=int, default=4, help='estimations run in parallel (default 4)')
    ap.add_argument('--force', action='store_true', help='rerun finished estimations')
    a = ap.parse_args()
    main(a.jobs, a.force)
