
"""
optimizer.py — Portfolio Optimization Core
============================================
Handles:
  - Fund returns & covariance estimation
  - Valuation-adjusted equity allocation (PE/PB z-score)
  - Constrained Markowitz optimization (portfolio vol <= 10%)
  - Vol constraint with 3-layer fallback chain
  - 10-year backtest (allocation + performance)

RETURN CALCULATION FIX (build_cov_from_returns)
------------------------------------------------
Old code:
    mean_ret = returns_df.mean().values * 252   # WRONG

This is the arithmetic annualized mean. It overstates the true compounded
annual growth rate (CAGR) by approximately vol^2 / 2 per year (Jensen's
inequality / Ito's lemma). For a fund with vol = 0.20, this overstates
returns by 2% per year. For vol = 0.25, by 3.1% per year.

When combined with the short-window bias (see data_fetcher.py), the
arithmetic method was producing 25-47% "returns" instead of realistic 11-16%.

Fix:
    Use the geometric (log-return) method:
        ann_ret_i = exp( mean(log(1 + r_daily_i)) * 252 ) - 1

    This is mathematically equivalent to:
        (final_NAV / initial_NAV)^(trading_days_per_year / total_days) - 1

    It correctly accounts for compounding and gives the true CAGR.
    The difference from arithmetic: exactly -vol^2/2 per year (removed bias).

COVARIANCE FIX
--------------
Old code used returns_df.cov() on a matrix where different funds had
different NaN patterns (due to different inception dates). pandas .cov()
uses pairwise complete observations by default, which is correct — it
uses the overlapping period for each pair. However, multiplying the
resulting matrix by 252 is correct only if returns are daily.

We keep returns_df.cov() * 252 but add a PSD regularisation step.

VOLATILITY
----------
Volatility computed from cov matrix diagonal is unaffected by the
arithmetic/geometric distinction — it was already correct.
"""

import numpy as np
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)

# ── GLOBAL PARAMETERS ────────────────────────────────────────────────────────
RF          = 0.065   # India 91-day T-bill
VOL_CAP     = 0.10    # Hard portfolio vol cap
VOL_TOL     = 0.0005  # Accept up to 10.05% (SLSQP numerical tolerance)
MIN_FUND_W  = 0.03
MAX_FUND_W  = 0.25

NIFTY_MEAN_PE, NIFTY_STD_PE = 22.0, 5.0
NIFTY_MEAN_PB, NIFTY_STD_PB = 3.2,  0.6

# Asset class parameters (annualized)
DEBT_RET, DEBT_VOL = 0.075, 0.040
GOLD_RET, GOLD_VOL = 0.100, 0.150
CASH_RET, CASH_VOL = 0.068, 0.005

RHO_EQ_DEBT   = -0.10
RHO_EQ_GOLD   =  0.05
RHO_DEBT_GOLD =  0.00

# Historical Nifty 50 PE/PB (Dec year-end snapshots)
NIFTY_HISTORY = {
    2015: {"pe": 22.0, "pb": 3.2},
    2016: {"pe": 22.7, "pb": 3.3},
    2017: {"pe": 26.4, "pb": 3.5},
    2018: {"pe": 24.0, "pb": 3.4},
    2019: {"pe": 28.5, "pb": 3.5},
    2020: {"pe": 37.5, "pb": 4.0},
    2021: {"pe": 27.5, "pb": 4.5},
    2022: {"pe": 22.3, "pb": 3.9},
    2023: {"pe": 23.7, "pb": 4.0},
    2024: {"pe": 22.0, "pb": 3.7},
}

FUNDS_DEFAULT = [
    "Parag Parikh", "Quant", "ICICI", "Motilal Oswal", "HDFC",
    "Aditya Birla", "ITI", "Tata", "Invesco", "Bank of India",
    "HSBC", "Edelweiss", "WhiteOak"
]

def _build_synthetic_params():
    """Reproducible synthetic params: ~13% geometric CAGR, 17-21% vol."""
    n = len(FUNDS_DEFAULT)
    # Realistic geometric CAGRs for Indian flexi-cap funds (long-run)
    ann_returns = np.array([0.14, 0.15, 0.12, 0.14, 0.13,
                             0.12, 0.13, 0.12, 0.12, 0.13,
                             0.12, 0.11, 0.14])
    np.random.seed(10)
    corr = np.full((n, n), 0.82)
    for i in range(n):
        for j in range(i + 1, n):
            corr[i, j] = corr[j, i] = 0.75 + 0.12 * np.random.random()
    np.fill_diagonal(corr, 1.0)
    vols = np.array([0.17, 0.21, 0.18, 0.20, 0.17,
                     0.18, 0.19, 0.18, 0.17, 0.20,
                     0.18, 0.17, 0.19])
    cov = np.outer(vols, vols) * corr
    cov = (cov + cov.T) / 2
    np.fill_diagonal(cov, vols ** 2)
    return ann_returns, cov, vols

_SYNTHETIC_RETURNS, _SYNTHETIC_COV, _SYNTHETIC_VOLS = _build_synthetic_params()


# ── COVARIANCE & RETURN ESTIMATION ───────────────────────────────────────────

def build_cov_from_returns(returns_df):
    """
    Build annualized geometric returns, covariance matrix, vols, and Sharpe
    ratios from a daily returns DataFrame.

    Returns DataFrame may have NaN for dates where a fund has no data
    (different inception dates). This is handled as follows:
      - Geometric return: computed per-fund on its own non-NaN observations
      - Covariance: pandas .cov() uses pairwise complete observations
        (default min_periods=1), which correctly uses overlapping periods
        for each pair of funds

    CRITICAL FIX: use geometric (log-return) annualization, NOT arithmetic.
        ann_ret_i = exp( mean(log(1 + r_i)) * 252 ) - 1
    This removes the +vol^2/2 upward bias in the arithmetic method.
    """
    import numpy as np

    n_funds = len(returns_df.columns)

    # ── Geometric annualized returns (per fund, on its own valid history) ──
    ann_returns = np.zeros(n_funds)
    n_obs       = np.zeros(n_funds, dtype=int)

    for i, col in enumerate(returns_df.columns):
        r = returns_df[col].dropna()  # use only this fund's valid days
        n_obs[i] = len(r)

        if len(r) < 2:
            logger.warning("Fund %s has only %d observations — defaulting to RF", col, len(r))
            ann_returns[i] = RF
            continue

        # Geometric CAGR via log-return method
        # log(1+r) → mean → * 252 → exp → -1
        # Numerically stable for small daily returns
        log_ret = np.log1p(r.values)
        ann_returns[i] = float(np.expm1(log_ret.mean() * 252))

        logger.info(
            "%-18s  n=%4d days  geo_CAGR=%.2f%%",
            col, len(r), ann_returns[i] * 100
        )

    # ── Annualized covariance matrix ──────────────────────────────────────
    # .cov() with default min_periods=1 uses pairwise overlapping periods.
    # Multiply by 252 to annualize (correct for daily returns).
    cov_mat = returns_df.cov().values * 252

    # Ensure positive semi-definite (floating point can introduce tiny
    # negative eigenvalues on short pairwise windows)
    eigvals = np.linalg.eigvalsh(cov_mat)
    if eigvals.min() < 0:
        cov_mat += (-eigvals.min() + 1e-8) * np.eye(n_funds)
        logger.debug("PSD fix applied: shift = %.2e", -eigvals.min() + 1e-8)

    # Annualized volatilities from diagonal
    vols = np.sqrt(np.diag(cov_mat))

    # Sharpe ratios (geometric return basis)
    fund_sharpes = (ann_returns - RF) / np.where(vols > 0, vols, np.nan)

    # Diagnostics
    logger.info(
        "Return range: %.1f%% – %.1f%%  (expected 10–18%% for Indian flexi-cap)",
        ann_returns.min() * 100, ann_returns.max() * 100
    )
    logger.info(
        "Vol range: %.1f%% – %.1f%%",
        vols.min() * 100, vols.max() * 100
    )
    logger.info(
        "Sharpe range: %.2f – %.2f  (expected 0.3–1.2)",
        np.nanmin(fund_sharpes), np.nanmax(fund_sharpes)
    )

    return ann_returns, cov_mat, vols, fund_sharpes


# ── VALUATION LOGIC ──────────────────────────────────────────────────────────

def valuation_z(pe, pb):
    z_pe = (pe - NIFTY_MEAN_PE) / NIFTY_STD_PE
    z_pb = (pb - NIFTY_MEAN_PB) / NIFTY_STD_PB
    return z_pe, z_pb, (z_pe + z_pb) / 2


def equity_from_z(z):
    """Map combined z-score → equity allocation in [50%, 90%]."""
    z_c = np.clip(z, -2.0, 2.0)
    return float(np.clip(0.70 - 0.10 * z_c, 0.50, 0.90))


def get_non_equity(E, z):
    """
    Split non-equity (1-E) into debt/gold/cash.
    Expensive market (z > 0): more debt+gold, less cash.
    Cheap market (z < 0): more cash, slightly less gold.
    """
    rem = 1.0 - E
    d = np.clip(0.55 + 0.05 * z, 0.30, 0.70)
    g = np.clip(0.30 + 0.05 * z, 0.10, 0.50)
    c = np.clip(0.15 - 0.10 * z, 0.05, 0.30)
    t = d + g + c
    return float(rem * d / t), float(rem * g / t), float(rem * c / t)


# ── PORTFOLIO STATS ──────────────────────────────────────────────────────────

def portfolio_vol_from_weights(E, w_eq, D, G, C, cov_mat):
    """
    Full portfolio volatility: equity sleeve + debt + gold + cash.

    Uses sigma_eq = E * sleeve_vol (portfolio-level equity vol block)
    so cross-asset covariance terms are correct.
    """
    sleeve_vol = float(np.sqrt(np.clip(w_eq @ cov_mat @ w_eq, 0, None)))
    sigma_eq   = E * sleeve_vol

    var = (
        sigma_eq ** 2 +
        D ** 2 * DEBT_VOL ** 2 +
        G ** 2 * GOLD_VOL ** 2 +
        C ** 2 * CASH_VOL ** 2 +
        2 * sigma_eq * D * DEBT_VOL * RHO_EQ_DEBT +
        2 * sigma_eq * G * GOLD_VOL * RHO_EQ_GOLD +
        2 * D * G * DEBT_VOL * GOLD_VOL * RHO_DEBT_GOLD
    )
    return float(np.sqrt(max(var, 0.0))), sleeve_vol


def portfolio_return(E, w_eq, D, G, C, ann_returns):
    eq_ret = float(np.dot(w_eq, ann_returns))
    return E * eq_ret + D * DEBT_RET + G * GOLD_RET + C * CASH_RET


def full_stats(E, w_eq, D, G, C, ann_returns, cov_mat):
    ret              = portfolio_return(E, w_eq, D, G, C, ann_returns)
    vol, sleeve_vol  = portfolio_vol_from_weights(E, w_eq, D, G, C, cov_mat)
    return ret, vol, sleeve_vol


# ── OPTIMIZATION ─────────────────────────────────────────────────────────────

def _min_vol_weights(n, E, D, G, C, cov_mat):
    """Fallback: find minimum-vol fund weights (ignores return objective)."""
    def obj(w):
        v, _ = portfolio_vol_from_weights(E, w, D, G, C, cov_mat)
        return v

    best = None
    for w0 in [np.ones(n) / n, np.random.dirichlet(np.ones(n) * 3)]:
        r = minimize(obj, w0, method='SLSQP',
                     bounds=[(MIN_FUND_W, MAX_FUND_W)] * n,
                     constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}],
                     options={'ftol': 1e-14, 'maxiter': 3000})
        if r.success:
            v, _ = portfolio_vol_from_weights(E, r.x, D, G, C, cov_mat)
            if best is None or v < best[1]:
                best = (r.x.copy(), v)

    return (best[0], best[1]) if best else (np.ones(n) / n, None)


def _scale_equity_to_meet_cap(w_eq, E_target, z, cov_mat, vol_cap=VOL_CAP):
    """
    If even minimum-vol weights violate the cap, step equity down in 0.5pp
    increments until satisfied. Bounded below by 50%.
    """
    E = E_target
    for _ in range(80):
        D, G, C = get_non_equity(E, z)
        v, _ = portfolio_vol_from_weights(E, w_eq, D, G, C, cov_mat)
        if v <= vol_cap + VOL_TOL:
            return E, D, G, C, v
        E = max(0.50, E - 0.005)

    D, G, C = get_non_equity(0.50, z)
    v, _ = portfolio_vol_from_weights(0.50, w_eq, D, G, C, cov_mat)
    return 0.50, D, G, C, v


def optimize_for_pe_pb(pe, pb, ann_returns, cov_mat, vols, fund_sharpes, funds,
                        vol_cap=VOL_CAP):
    """
    Main optimization entry point.

    Maximise portfolio return subject to:
      - portfolio vol <= vol_cap (10%)
      - fund weights in [MIN_FUND_W, MAX_FUND_W] (3%–25%)
      - weights sum to 1
      - equity allocation fixed by PE/PB valuation z-score

    Fallback chain if primary optimizer fails:
      1. Re-run with 10 random starting points
      2. Use minimum-vol weights
      3. Scale equity down until cap is satisfied
    """
    n = len(funds)
    z_pe, z_pb, z = valuation_z(pe, pb)
    E_val = equity_from_z(z)
    D0, G0, C0 = get_non_equity(E_val, z)

    def neg_ret(w):
        return -portfolio_return(E_val, w, D0, G0, C0, ann_returns)

    def vol_constraint(w):
        v, _ = portfolio_vol_from_weights(E_val, w, D0, G0, C0, cov_mat)
        return vol_cap - v   # >= 0 for SLSQP ineq

    rng   = np.random.default_rng(int(pe * 100 + pb * 10))
    inits = ([np.ones(n) / n] +
             [rng.dirichlet(np.ones(n) * 2) for _ in range(9)])

    best_ret, best_w, best_vol = -np.inf, None, None

    for w0 in inits:
        w0 = np.asarray(w0, dtype=float)
        w0 /= w0.sum()
        try:
            r = minimize(
                neg_ret, w0, method='SLSQP',
                bounds=[(MIN_FUND_W, MAX_FUND_W)] * n,
                constraints=[
                    {'type': 'eq',   'fun': lambda w: np.sum(w) - 1.0},
                    {'type': 'ineq', 'fun': vol_constraint},
                ],
                options={'ftol': 1e-13, 'maxiter': 5000},
            )
        except Exception:
            continue

        if not r.success:
            continue

        w_cand = np.clip(r.x, MIN_FUND_W, MAX_FUND_W)
        w_cand /= w_cand.sum()
        v_cand, _ = portfolio_vol_from_weights(E_val, w_cand, D0, G0, C0, cov_mat)
        ret_cand   = portfolio_return(E_val, w_cand, D0, G0, C0, ann_returns)

        if v_cand <= vol_cap + VOL_TOL and ret_cand > best_ret:
            best_ret, best_w, best_vol = ret_cand, w_cand.copy(), v_cand

    # Fallback: min-vol weights
    if best_w is None:
        logger.warning("Primary optimizer failed for PE=%.1f PB=%.1f — min-vol fallback", pe, pb)
        best_w, best_vol = _min_vol_weights(n, E_val, D0, G0, C0, cov_mat)
        best_ret = portfolio_return(E_val, best_w, D0, G0, C0, ann_returns)

    # Hard validation: scale equity if still over cap
    v_final, sl_vol = portfolio_vol_from_weights(E_val, best_w, D0, G0, C0, cov_mat)
    E_final, D_final, G_final, C_final = E_val, D0, G0, C0

    if v_final > vol_cap + VOL_TOL:
        logger.warning("Vol %.4f > cap at PE=%.1f — scaling equity down", v_final, pe)
        E_final, D_final, G_final, C_final, v_final = _scale_equity_to_meet_cap(
            best_w, E_val, z, cov_mat, vol_cap
        )

    ret_final  = portfolio_return(E_final, best_w, D_final, G_final, C_final, ann_returns)
    _, sl_vol  = portfolio_vol_from_weights(E_final, best_w, D_final, G_final, C_final, cov_mat)
    sharpe     = float((ret_final - RF) / v_final) if v_final > 0 else 0.0

    return {
        "pe": pe, "pb": pb,
        "z_pe": round(z_pe, 4), "z_pb": round(z_pb, 4), "z": round(z, 4),
        "equity":    round(E_final, 4),
        "debt":      round(D_final, 4),
        "gold":      round(G_final, 4),
        "cash":      round(C_final, 4),
        "port_ret":  round(ret_final, 4),
        "port_vol":  round(v_final, 4),
        "sharpe":    round(sharpe, 4),
        "eq_vol":    round(sl_vol, 4),
        "fund_weights":  [round(float(w), 4) for w in best_w],
        "fund_sharpes":  [round(float(s), 3) for s in fund_sharpes],
        "fund_returns":  [round(float(r), 4) for r in ann_returns],
        "fund_vols":     [round(float(v), 4) for v in vols],
        "constraint_ok": bool(v_final <= vol_cap + VOL_TOL),
    }


# ── EFFICIENT FRONTIER ───────────────────────────────────────────────────────

def compute_frontier(ann_returns, cov_mat, z=0.25, n_points=12):
    """
    Trace efficient frontier by sweeping equity from 50% to 90%.
    For each level, maximise return (unconstrained by vol cap for shape).
    """
    n = len(ann_returns)
    frontier = []
    for E_fix in np.linspace(0.50, 0.90, n_points):
        D, G, C = get_non_equity(float(E_fix), z)

        def neg_r(w):
            return -portfolio_return(float(E_fix), w, D, G, C, ann_returns)

        r = minimize(neg_r, np.ones(n) / n, method='SLSQP',
                     bounds=[(MIN_FUND_W, MAX_FUND_W)] * n,
                     constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}],
                     options={'ftol': 1e-12, 'maxiter': 2000})
        if r.success:
            v, _ = portfolio_vol_from_weights(float(E_fix), r.x, D, G, C, cov_mat)
            ret   = portfolio_return(float(E_fix), r.x, D, G, C, ann_returns)
            frontier.append({"v": round(v, 4), "r": round(ret, 4), "E": round(float(E_fix), 2)})

    return sorted(frontier, key=lambda x: x["v"])


# ── ALLOCATION BACKTEST ───────────────────────────────────────────────────────

def run_backtest(ann_returns, cov_mat, vols, fund_sharpes, funds,
                 vol_cap=VOL_CAP, history=None):
    """
    Year-by-year allocation backtest using historical Nifty PE/PB snapshots.
    Returns allocation history (NOT realised performance).
    """
    if history is None:
        history = NIFTY_HISTORY

    results = []
    for year in sorted(history.keys()):
        pe = history[year]["pe"]
        pb = history[year]["pb"]
        res = optimize_for_pe_pb(
            pe, pb, ann_returns, cov_mat, vols, fund_sharpes, funds, vol_cap
        )
        fw_sorted = sorted(zip(funds, res["fund_weights"]), key=lambda x: -x[1])[:3]
        results.append({
            "year":          year,
            "pe":            pe,
            "pb":            pb,
            "z":             res["z"],
            "equity":        res["equity"],
            "debt":          res["debt"],
            "gold":          res["gold"],
            "cash":          res["cash"],
            "port_ret":      res["port_ret"],
            "port_vol":      res["port_vol"],
            "sharpe":        res["sharpe"],
            "constraint_ok": res["constraint_ok"],
            "top_funds":     [{"name": f, "weight": round(w, 4)} for f, w in fw_sorted],
        })
    return results


# ── PERFORMANCE BACKTEST ──────────────────────────────────────────────────────

# Historical Nifty 50 TRI annual returns (approximate, Dec–Dec)
_NIFTY_ANNUAL_RETURNS = {
    2015: -0.040, 2016:  0.033, 2017:  0.288, 2018:  0.033, 2019:  0.122,
    2020:  0.147, 2021:  0.242, 2022:  0.043, 2023:  0.197, 2024:  0.088,
}

# Historical Gold (INR) annual returns (approximate)
_GOLD_ANNUAL_RETURNS = {
    2015: -0.060, 2016:  0.110, 2017:  0.050, 2018:  0.075, 2019:  0.190,
    2020:  0.280, 2021: -0.040, 2022:  0.110, 2023:  0.130, 2024:  0.210,
}

_DEBT_ANNUAL_RETURNS = {y: 0.068 for y in range(2015, 2025)}
_CASH_ANNUAL_RETURNS = {y: 0.065 for y in range(2015, 2025)}
_NIFTY_LONGRUN       = 0.12


def run_performance_backtest(ann_returns, cov_mat, vols, fund_sharpes, funds,
                              vol_cap=VOL_CAP, history=None):
    """
    True performance backtest: applies realised asset-class returns to the
    valuation-optimal allocation for each year, and compounds into a
    portfolio value series starting at 100.

    Equity sleeve return estimation:
        fund_actual_ret_i = Nifty_ret + (geometric_CAGR_i - nifty_longrun)

    This uses each fund's historical alpha over the long-run market average
    and applies it to the realised Nifty return for each year.
    """
    if history is None:
        history = NIFTY_HISTORY

    years_sorted   = sorted(history.keys())
    n_years        = len(years_sorted)
    fund_alphas    = ann_returns - _NIFTY_LONGRUN

    portfolio_value   = 100.0
    value_series      = [100.0]
    annual_ret_series = []
    per_year          = []

    for year in years_sorted:
        pe = history[year]["pe"]
        pb = history[year]["pb"]
        res = optimize_for_pe_pb(
            pe, pb, ann_returns, cov_mat, vols, fund_sharpes, funds, vol_cap
        )

        E  = res["equity"]
        D  = res["debt"]
        G  = res["gold"]
        C  = res["cash"]
        fw = np.array(res["fund_weights"])

        nifty_ret     = _NIFTY_ANNUAL_RETURNS.get(year, _NIFTY_LONGRUN)
        fund_rets     = nifty_ret + fund_alphas
        eq_sleeve_ret = float(np.dot(fw, fund_rets))
        debt_ret      = _DEBT_ANNUAL_RETURNS.get(year, DEBT_RET)
        gold_ret      = _GOLD_ANNUAL_RETURNS.get(year, GOLD_RET)
        cash_ret      = _CASH_ANNUAL_RETURNS.get(year, CASH_RET)

        port_ret = E*eq_sleeve_ret + D*debt_ret + G*gold_ret + C*cash_ret
        portfolio_value *= (1.0 + port_ret)

        value_series.append(round(portfolio_value, 4))
        annual_ret_series.append(round(port_ret, 6))
        per_year.append({
            "year":       year, "pe": pe, "pb": pb,
            "equity":     round(E, 4), "debt": round(D, 4),
            "gold":       round(G, 4), "cash": round(C, 4),
            "nifty_ret":  round(nifty_ret, 4),
            "port_ret":   round(port_ret, 6),
            "port_value": round(portfolio_value, 4),
        })

    rets_arr  = np.array(annual_ret_series)
    cagr      = float((portfolio_value / 100.0) ** (1.0 / n_years) - 1.0)
    vol_bt    = float(np.std(rets_arr, ddof=1))
    sharpe_bt = float((cagr - RF) / vol_bt) if vol_bt > 0 else 0.0

    peak = 100.0
    max_dd = 0.0
    for v in value_series[1:]:
        peak = max(peak, v)
        dd   = (v - peak) / peak
        if dd < max_dd:
            max_dd = dd

    return {
        "years":           [y for y in years_sorted],
        "portfolio_value": value_series[1:],
        "initial_value":   100.0,
        "annual_returns":  annual_ret_series,
        "cagr":            round(cagr, 6),
        "max_drawdown":    round(max_dd, 6),
        "volatility":      round(vol_bt, 6),
        "sharpe":          round(sharpe_bt, 4),
        "per_year":        per_year,
    }


# ── PUBLIC API ────────────────────────────────────────────────────────────────

def get_optimizer_state(funds=None, ann_returns=None, cov_mat=None,
                         vols=None, fund_sharpes=None):
    """
    Returns full optimizer state dict. Called by app.py on startup.
    Falls back to synthetic params if live data is unavailable.
    """
    if funds is None:
        funds        = FUNDS_DEFAULT
        ann_returns  = _SYNTHETIC_RETURNS
        cov_mat      = _SYNTHETIC_COV
        vols         = _SYNTHETIC_VOLS
        fund_sharpes = (ann_returns - RF) / vols

    return {
        "funds":        funds,
        "ann_returns":  ann_returns,
        "cov_mat":      cov_mat,
        "vols":         vols,
        "fund_sharpes": fund_sharpes,
    }
