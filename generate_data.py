"""
generate_data.py: step 1. Generates the synthetic panel data of Section 4.1 and Appendix A.1.

Each driver n has characteristics z_n = (INCOME, TOD, DUR) and one taste vector beta_n. The driver makes
T = 25 choices among J = 3 unlabeled stations (no alternative-specific constants):

    V_nit = beta_TT,n * TT_nit + beta_POI,n * POI_nit - CO_nit                              (Eq. 13)

The cost coefficient is fixed at -1, so the tastes are in $ per minute / $ per POI. Each taste is a
systematic part f_k(z) plus a skewed random component eta (Eqs. 14-15, A.1):

    beta_k,n = f_k(z_n) + sigma * s_k(z_n) * u_k,n
    f_k(z)   = phi(z)' c_k,   s_k(z) = sqrt(sum_m c_km^2 phi_m(z)^2),
    phi(z)   = (1, inc, tod, dur, inc*tod, inc*dur, tod*dur),
    inc = (ln INCOME - ln 60) / 0.35,  tod = TOD,  dur = (DUR - 60) / 30.

u_TT = 1 - G and u_POI = G - 1 with G ~ Exponential(1). Both have mean 0 and variance 1, with skewness
-2 and +2. u is drawn as the quantile transform of a standard normal variable, so the gamma shape
SKEW = 1 gives the exponential. The attributes are drawn anew on every occasion:
TT ~ U(0, 10), POI ~ U(0, 20), CO ~ U(0, 6).

Occasions t = 1..20 form the history pool, and a dataset with C choices per driver uses t <= C.
Occasions t = 21..25 are the five evaluation choices, the same for every C. The 2,000 drivers are cut
into consecutive blocks of 1,200 training, 400 development and 400 test drivers. The drivers are
i.i.d., so this is a random split. Both noise levels use the same seed: z, the attributes and the
choice uniforms are identical, and only sigma differs.

Output (data/):
    {split}_{tag}.csv            one row per choice occasion:
                                 iid, t, choice_id, INCOME, TOD, DUR, CHOICE, ALT{j}_{CO,TT,POI,AV}
    {split}_{tag}_truebeta.csv   one row per driver:
                                 iid, INCOME, TOD, DUR, beta_TT, beta_POI, beta_CO, f_TT, f_POI
with split in {train, dev, test} and tag in {noise20, noise50}. iid counts from 0 in each file.

Usage: python generate_data.py
"""
import numpy as np
import pandas as pd
from scipy.stats import gamma as gamma_dist, norm

import config as cfg

J, T = cfg.J, cfg.T
COST_COEF = -1.0
SKEW = 1.0                              # gamma shape of u (skewness 2 / sqrt(SKEW)); 1 = exponential
SKEW_SIGN = {'TT': -1, 'POI': 1}        # direction of the long tail

# taste coefficients c_k on phi(z) = [1, inc, tod, dur, inc*tod, inc*dur, tod*dur]   (Eqs. 14-15)
COEF = {'TT':  np.array([-0.50, -0.20, -0.10, 0.15, -0.05, 0.05, -0.10]),
        'POI': np.array([0.30, 0.10, -0.20, 0.25, -0.05, 0.10, -0.10])}
TASTES = list(COEF)

TT_MIN, TT_MAX = 0.0, 10.0              # travel time (min)
POI_MIN, POI_MAX = 0.0, 20.0            # points of interest near the station
CO_MIN, CO_MAX = 0.0, 6.0               # charging cost ($)
INC_MEAN_LOG, INC_SD_LOG = np.log(60.0), 0.35     # annual income ($1000), median 60


def standardize_z(income, tod, dur):
    """Raw (INCOME, TOD, DUR) -> z = (inc, tod, dur) of the taste functions."""
    inc = (np.log(income) - INC_MEAN_LOG) / INC_SD_LOG
    return np.column_stack([inc, np.asarray(tod, float), (np.asarray(dur, float) - 60.0) / 30.0])


def phi(Z):
    inc, tod, dur = Z.T
    return np.column_stack([np.ones(len(Z)), inc, tod, dur, inc * tod, inc * dur, tod * dur])


def taste_sd(F):
    """s_k(z) of every driver, (n, 2)."""
    return np.column_stack([np.sqrt((F ** 2) @ (COEF[t] ** 2)) for t in TASTES])


def standardized_u(Z):
    """u (mean 0, variance 1) from standard normal Z (n, 2) by the rank-preserving quantile transform:
    gamma with shape SKEW, long tail in the direction SKEW_SIGN."""
    u = np.empty_like(Z)
    for k, t in enumerate(TASTES):
        d = SKEW_SIGN[t]
        tail = norm.sf(Z[:, k]) if d > 0 else norm.cdf(Z[:, k])
        u[:, k] = d * (gamma_dist.isf(tail, SKEW) - SKEW) / np.sqrt(SKEW)
    return u


def generate(n, sigma, seed, occasions=T):
    """n drivers x `occasions` choices -> long panel, true tastes, true choice probabilities."""
    rng = np.random.default_rng(seed)

    # driver characteristics z and tastes
    income = rng.lognormal(INC_MEAN_LOG, INC_SD_LOG, size=n)
    tod = rng.integers(0, 2, size=n)                          # 0 day, 1 night
    dur = rng.choice([30, 60, 90], size=n)                    # usual charging duration (min)
    F = phi(standardize_z(income, tod, dur))                  # (n, 7)
    xi = rng.standard_normal((n, len(TASTES), F.shape[1]))    # identical at every noise level
    f = np.column_stack([F @ COEF[t] for t in TASTES])        # systematic tastes f_k(z)
    s = taste_sd(F)
    Z = np.column_stack([(F * np.abs(COEF[t]) * xi[:, k]).sum(1) for k, t in enumerate(TASTES)]) / s
    beta = f + sigma * s * standardized_u(Z)                  # beta_n = f(z_n) + eta_n

    # choice occasions: new attributes every time, logit choice with the driver's tastes
    TT, POI, CO = (np.empty((n, occasions, J)) for _ in range(3))
    P, choice = np.empty((n, occasions, J)), np.empty((n, occasions), dtype=int)
    for t in range(occasions):
        TT[:, t] = rng.uniform(TT_MIN, TT_MAX, size=(n, J))
        POI[:, t] = rng.uniform(POI_MIN, POI_MAX, size=(n, J))
        CO[:, t] = rng.uniform(CO_MIN, CO_MAX, size=(n, J))
        V = beta[:, [0]] * TT[:, t] + beta[:, [1]] * POI[:, t] + COST_COEF * CO[:, t]
        P[:, t] = np.exp(V - V.max(1, keepdims=True))
        P[:, t] /= P[:, t].sum(1, keepdims=True)
        choice[:, t] = np.minimum((rng.random(n)[:, None] > P[:, t].cumsum(1)).sum(1), J - 1)

    iid = np.repeat(np.arange(n), occasions)
    df = pd.DataFrame({'iid': iid, 't': np.tile(np.arange(1, occasions + 1), n),
                       'choice_id': np.arange(n * occasions),
                       'INCOME': income[iid], 'TOD': tod[iid], 'DUR': dur[iid], 'CHOICE': choice.ravel()})
    for j in range(J):
        df[f'ALT{j}_CO'] = CO[:, :, j].ravel()
        df[f'ALT{j}_TT'] = TT[:, :, j].ravel()
        df[f'ALT{j}_POI'] = POI[:, :, j].ravel()
        df[f'ALT{j}_AV'] = 1
    truth = pd.DataFrame({'iid': np.arange(n), 'INCOME': income, 'TOD': tod, 'DUR': dur,
                          'beta_TT': beta[:, 0], 'beta_POI': beta[:, 1], 'beta_CO': COST_COEF,
                          'f_TT': f[:, 0], 'f_POI': f[:, 1]})
    return df, truth, P.reshape(n * occasions, J)


def split(df, truth, P):
    """Cut the sample into the consecutive blocks of cfg.SPLIT -> {split: (df, truth, P)}, ids from 0."""
    out, start = {}, 0
    for name, n in cfg.SPLIT.items():
        occ = ((df['iid'] >= start) & (df['iid'] < start + n)).values
        d = df[occ].reset_index(drop=True)
        t = truth.iloc[start:start + n].reset_index(drop=True)
        d['iid'] -= start
        t['iid'] -= start
        d['choice_id'] = np.arange(len(d))
        out[name], start = (d, t, P[occ]), start + n
    return out


def main():
    cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
    for tag, sigma in cfg.NOISE.items():
        for name, (df, truth, P) in split(*generate(cfg.N_DRIVERS, sigma, cfg.SEED)).items():
            df.to_csv(cfg.DATA_DIR / f'{name}_{tag}.csv', index=False)
            truth.to_csv(cfg.DATA_DIR / f'{name}_{tag}_truebeta.csv', index=False)
            b = truth[['beta_TT', 'beta_POI']].values
            print(f'[sigma = {sigma}, {name:5s}] {len(truth):4d} drivers x {T} occasions | '
                  f'mean beta TT/POI = {np.round(b.mean(0), 3)}, SD = {np.round(b.std(0), 3)}')
    print(f'written -> {cfg.DATA_DIR}')


if __name__ == '__main__':
    main()
