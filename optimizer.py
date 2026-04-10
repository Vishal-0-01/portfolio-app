"""
optimizer.py — Portfolio Optimization Core
============================================
Handles:
  - Fund returns & covariance estimation
  - Valuation-adjusted equity allocation (PE/PB z-score)
  - Constrained Markowitz optimization (portfolio vol ≤ 10%)
  - Vol constraint bug fixes (post-optimization validation + fallback)
  - 10-year backtest simulation
"""

import numpy as np
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)

# ── GLOBAL PARAMETERS ────────────────────────────────────────────────────────
RF          = 0.065   # India 91-day T-bill
VOL_CAP     = 0.10    # Hard portfolio vol cap
VOL_TOL     = 0.0005  # Tolerance: accept up to 10.05% (optimizer numerical error)
MIN_FUND_W  = 0.03
MAX_FUND_W  = 0.25

NIFTY_MEAN_PE, NIFTY_STD_PE = 22.0, 5.0
NIFTY_MEAN_PB, NIFTY_STD_PB = 3.2,  0.6

# Asset class parameters (annualized)
DEBT_RET, DEBT_VOL = 0.075, 0.040
GOLD_RET, GOLD_VOL = 0.100, 0.150
CASH_RET, CASH_VOL = 0.068, 0.005

# Cross-asset correlations (equity with others)
RHO_EQ_DEBT   = -0.10
RHO_EQ_GOLD   =  0.05
RHO_DEBT_GOLD =  0.00

# Static 10-year Nifty PE/PB history (annual snapshots, Dec year-end)
# Sources: NSE historical P/E data
NIFTY_HISTORY = {
    2015: {"pe": 22.0, "pb": 3.2},
    2016: {"pe": 22.7, "pb": 3.3},
    2017: {"pe": 26.4, "pb": 3.5},
    2018: {"pe": 24.0, "pb": 3.4},
    2019: {"pe": 28.5, "pb": 3.5},
    2020: {"pe": 37.5, "pb": 4.0},   # COVID recovery spike
    2021: {"pe": 27.5, "pb": 4.5},
    2022: {"pe": 22.3, "pb": 3.9},
    2023: {"pe": 23.7, "pb": 4.0},
    2024: {"pe": 22.0, "pb": 3.7},
}

# Synthetic returns & cov used when mftool is unavailable
# (replace with live data via build_cov_from_returns() after fetching NAVs)
FUNDS_DEFAULT = [
    "Parag Parikh", "Quant", "ICICI", "Motilal Oswal", "HDFC",
    "Aditya Birla", "ITI", "Tata", "Invesco", "Bank of India",
    "HSBC", "Edelweiss", "WhiteOak"
]

def _build_synthetic_params(seed=42):
    """Reproducible synthetic params for demo / fallback."""
    n = len(FUNDS_DEFAULT)
    ann_returns = np.array([0.18, 0.22, 0.16, 0.20, 0.17,
                             0.15, 0.19, 0.16, 0.15, 0.20,
                             0.16, 0.14, 0.21])
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


# ── COVARIANCE HELPERS ───────────────────────────────────────────────────────

def build_cov_from_returns(returns_df):
    """
    Build annualized mean returns, covariance matrix, vols, and Sharpe ratios
    from a daily returns DataFrame.

    FIXED:
    - Uses log returns (more stable than simple mean)
    - Clips extreme daily returns (removes spikes)
    - Improves covariance stability
    """

    # ── Step 1: Clean data ─────────────────────────────
    returns_df = returns_df.copy()

    # Remove extreme outliers (bad NAV jumps / bad data)
    returns_df = returns_df.clip(lower=-0.20, upper=0.20)

    # Drop any remaining NaNs
    returns_df = returns_df.dropna()

    # ── Step 2: Log returns for stability ──────────────
    log_returns = np.log1p(returns_df)

    # Annualized return (log → exp)
    mean_log = log_returns.mean().values
    ann_returns = np.exp(mean_log * 252) - 1

    # ── Step 3: Covariance ─────────────────────────────
    cov_mat = log_returns.cov().values * 252

    # Ensure PSD (numerical stability)
    eigvals = np.linalg.eigvalsh(cov_mat)
    if eigvals.min() < 0:
        cov_mat += (-eigvals.min() + 1e-8) * np.eye(len(mean_log))

    # ── Step 4: Volatility ─────────────────────────────
    vols = np.sqrt(np.diag(cov_mat))

    # ── Step 5: Sharpe ratio ───────────────────────────
    sharpe = np.zeros_like(vols)
    valid = vols > 1e-8
    sharpe[valid] = (ann_returns[valid] - RF) / vols[valid]

    return ann_returns, cov_mat, vols, sharpe

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
    Split non-equity (1-E) into debt/gold/cash proportions.
    When market is expensive (z > 0): more debt+gold, less cash.
    When cheap (z < 0): more cash, slightly less gold.
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
    Full portfolio volatility including equity sleeve + debt + gold + cash.

    BUG ROOT CAUSE (FIXED HERE):
    The original code computed equity volatility as sqrt(w @ cov @ w) where
    cov was already annualised. This is correct. However, the variance
    computation used:

        2*E*D * eq_vol * DEBT_VOL * RHO_EQ_DEBT

    which gives the covariance term between the PORTFOLIO-LEVEL equity block
    and the debt block. This is correct IF eq_vol is the *portfolio-level*
    equity vol (i.e., E * sleeve_vol). The original code used sleeve_vol,
    not portfolio-level vol — causing the cross-terms to be underestimated
    and allowing the total var to appear lower than it really is.

    Fix: factor E into the covariance cross-terms correctly, i.e. use
    the actual portfolio dollar-weights: w_portfolio_eq = E * w_eq.
    Then sigma_eq = sqrt((E*w_eq)' Cov (E*w_eq)) = E * sleeve_vol.
    """
    sleeve_vol = float(np.sqrt(np.clip(w_eq @ cov_mat @ w_eq, 0, None)))
    sigma_eq   = E * sleeve_vol   # ← portfolio-level equity vol block

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
    ret  = portfolio_return(E, w_eq, D, G, C, ann_returns)
    vol, sleeve_vol = portfolio_vol_from_weights(E, w_eq, D, G, C, cov_mat)
    return ret, vol, sleeve_vol


# ── OPTIMIZATION ─────────────────────────────────────────────────────────────

def _min_vol_weights(n, E, D, G, C, cov_mat, vol_cap=VOL_CAP):
    """
    Fallback: find minimum-vol fund weights that satisfy the portfolio vol cap.
    Used when the max-return optimizer fails or violates the constraint.
    """
    def obj(w):
        v, _ = portfolio_vol_from_weights(E, w, D, G, C, cov_mat)
        return v

    best = None
    for w0 in [np.ones(n) / n,
               np.random.dirichlet(np.ones(n) * 3)]:
        r = minimize(obj, w0, method='SLSQP',
                     bounds=[(MIN_FUND_W, MAX_FUND_W)] * n,
                     constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}],
                     options={'ftol': 1e-14, 'maxiter': 3000})
        if r.success:
            v, _ = portfolio_vol_from_weights(E, r.x, D, G, C, cov_mat)
            if best is None or v < best[1]:
                best = (r.x, v)

    if best is None:
        return np.ones(n) / n, None
    return best[0], best[1]


def _scale_equity_to_meet_cap(w_eq, E_target, D_base, G_base, C_base, z,
                               cov_mat, vol_cap=VOL_CAP):
    """
    If even minimum-vol fund weights still breach the cap at E_target,
    step E down (within [50%, 90%]) until the cap is satisfied.
    Returns (E_final, D, G, C) that meet vol_cap.
    """
    E = E_target
    for _ in range(100):
        D, G, C = get_non_equity(E, z)
        v, _ = portfolio_vol_from_weights(E, w_eq, D, G, C, cov_mat)
        if v <= vol_cap + VOL_TOL:
            return E, D, G, C, v
        E = max(0.50, E - 0.005)   # reduce equity by 0.5pp steps

    # Last resort: equity = 50%
    D, G, C = get_non_equity(0.50, z)
    v, _ = portfolio_vol_from_weights(0.50, w_eq, D, G, C, cov_mat)
    return 0.50, D, G, C, v


def optimize_for_pe_pb(pe, pb, ann_returns, cov_mat, vols, fund_sharpes, funds,
                        vol_cap=VOL_CAP):
    """
    Main entry point. Returns full allocation dict for given PE/PB.

    FIXES applied:
    1. Correct portfolio vol formula (E * sleeve_vol in cross-terms).
    2. Post-optimization hard validation: if result > vol_cap + VOL_TOL,
       trigger fallback chain.
    3. Fallback chain:
       a) Re-run optimizer with tighter tolerance and more random starts.
       b) Use min-vol weights.
       c) Scale down equity allocation until cap is met.
    4. All random seeds fixed per (pe, pb) so results are reproducible.
    """
    n = len(funds)
    z_pe, z_pb, z = valuation_z(pe, pb)
    E_val = equity_from_z(z)
    D0, G0, C0 = get_non_equity(E_val, z)

    # ── Step 1: max-return optimizer ─────────────────────────────────────
    def neg_ret(w):
        return -portfolio_return(E_val, w, D0, G0, C0, ann_returns)

    def vol_constraint(w):
        # SLSQP ineq: must be >= 0  →  vol_cap - vol >= 0
        v, _ = portfolio_vol_from_weights(E_val, w, D0, G0, C0, cov_mat)
        return vol_cap - v

    rng = np.random.default_rng(int(pe * 100 + pb * 10))
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
        w_cand /= w_cand.sum()          # re-normalise after clipping
        v_cand, _ = portfolio_vol_from_weights(E_val, w_cand, D0, G0, C0, cov_mat)
        ret_cand   = portfolio_return(E_val, w_cand, D0, G0, C0, ann_returns)

        if v_cand <= vol_cap + VOL_TOL and ret_cand > best_ret:
            best_ret, best_w, best_vol = ret_cand, w_cand.copy(), v_cand

    # ── Step 2: fallback — min-vol weights ───────────────────────────────
    if best_w is None:
        logger.warning("Optimizer failed for PE=%.1f PB=%.1f — using min-vol fallback", pe, pb)
        best_w, best_vol = _min_vol_weights(n, E_val, D0, G0, C0, cov_mat, vol_cap)
        best_ret = portfolio_return(E_val, best_w, D0, G0, C0, ann_returns)

    # ── Step 3: hard validation — scale equity if still over cap ─────────
    v_final, sl_vol = portfolio_vol_from_weights(E_val, best_w, D0, G0, C0, cov_mat)
    E_final, D_final, G_final, C_final = E_val, D0, G0, C0

    if v_final > vol_cap + VOL_TOL:
        logger.warning("Vol %.4f > cap %.4f at PE=%.1f — scaling equity down", v_final, vol_cap, pe)
        E_final, D_final, G_final, C_final, v_final = _scale_equity_to_meet_cap(
            best_w, E_val, D0, G0, C0, z, cov_mat, vol_cap
        )

    ret_final  = portfolio_return(E_final, best_w, D_final, G_final, C_final, ann_returns)
    _, sl_vol  = portfolio_vol_from_weights(E_final, best_w, D_final, G_final, C_final, cov_mat)
    sharpe     = (ret_final - RF) / v_final if v_final > 0 else 0.0

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
    For each equity level: maximise return (unconstrained by vol cap,
    so we can show the full frontier shape), then record (vol, ret, E).
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
            frontier.append({
                "v": round(v, 4),
                "r": round(ret, 4),
                "E": round(float(E_fix), 2),
            })

    return sorted(frontier, key=lambda x: x["v"])


# ── BACKTEST ─────────────────────────────────────────────────────────────────

def run_backtest(ann_returns, cov_mat, vols, fund_sharpes, funds,
                 vol_cap=VOL_CAP, history=None):
    """
    Simulate year-by-year allocation from 2015–2024 using historical
    Nifty PE/PB snapshots.

    Returns list of dicts, one per year, suitable for time-series charting.
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

        # Top 3 fund weights for display
        fw = list(zip(funds, res["fund_weights"]))
        fw_sorted = sorted(fw, key=lambda x: -x[1])[:3]

        results.append({
            "year":        year,
            "pe":          pe,
            "pb":          pb,
            "z":           res["z"],
            "equity":      res["equity"],
            "debt":        res["debt"],
            "gold":        res["gold"],
            "cash":        res["cash"],
            "port_ret":    res["port_ret"],
            "port_vol":    res["port_vol"],
            "sharpe":      res["sharpe"],
            "constraint_ok": res["constraint_ok"],
            "top_funds":   [{"name": f, "weight": round(w, 4)} for f, w in fw_sorted],
        })

    return results


# ── PERFORMANCE BACKTEST ─────────────────────────────────────────────────────

# Historical Nifty 50 TRI annual returns (approximate, Dec–Dec)
# Sources: NSE India, moneycontrol historical data
_NIFTY_ANNUAL_RETURNS = {
    2015: -0.040,   # -4.1%  (bear year)
    2016:  0.033,   #  3.3%
    2017:  0.288,   # 28.8%  (bull run)
    2018:  0.033,   #  3.3%  (flat/volatile)
    2019:  0.122,   # 12.2%
    2020:  0.147,   # 14.7%  (crash + sharp recovery)
    2021:  0.242,   # 24.2%  (post-COVID rally)
    2022:  0.043,   #  4.3%  (Russia/rate-hike year)
    2023:  0.197,   # 19.7%
    2024:  0.088,   #  8.8%  (estimate through Dec)
}

# Historical Gold (INR) annual returns (approximate)
_GOLD_ANNUAL_RETURNS = {
    2015: -0.060,
    2016:  0.110,
    2017:  0.050,
    2018:  0.075,
    2019:  0.190,
    2020:  0.280,
    2021: -0.040,
    2022:  0.110,
    2023:  0.130,
    2024:  0.210,
}

# Debt returns: approximate short-duration/liquid fund returns
_DEBT_ANNUAL_RETURNS = {y: 0.068 for y in range(2015, 2025)}
_CASH_ANNUAL_RETURNS = {y: 0.065 for y in range(2015, 2025)}

# Long-run Nifty return used for alpha decomposition
_NIFTY_LONGRUN = 0.12


def run_performance_backtest(ann_returns, cov_mat, vols, fund_sharpes, funds,
                              vol_cap=VOL_CAP, history=None):
    """
    True performance backtest: for each year, get the valuation-optimal
    allocation then apply that year's realised asset-class returns to build
    a compounding portfolio value series.

    Equity sleeve return estimation:
        fund_actual_ret_i = Nifty_ret + (ann_return_i - nifty_longrun)
    This uses each fund's historical alpha over the market (derived from the
    same covariance/return data already in use) and applies it to the
    realised Nifty return each year. This keeps the model internally
    consistent without requiring separate NAV history.

    Does NOT modify any existing optimizer functions.
    """
    if history is None:
        history = NIFTY_HISTORY

    years_sorted = sorted(history.keys())
    n_years      = len(years_sorted)

    # Fund alpha over market (stable characteristic, computed once)
    fund_alphas = ann_returns - _NIFTY_LONGRUN   # shape (n_funds,)

    portfolio_value = 100.0
    value_series    = [100.0]   # starts at 100 at beginning of 2015
    annual_ret_series = []
    per_year = []

    for year in years_sorted:
        pe = history[year]["pe"]
        pb = history[year]["pb"]

        res = optimize_for_pe_pb(
            pe, pb, ann_returns, cov_mat, vols, fund_sharpes, funds, vol_cap
        )

        E   = res["equity"]
        D   = res["debt"]
        G   = res["gold"]
        C   = res["cash"]
        fw  = np.array(res["fund_weights"])

        # Realised returns for this year
        nifty_ret     = _NIFTY_ANNUAL_RETURNS.get(year, _NIFTY_LONGRUN)
        fund_rets     = nifty_ret + fund_alphas          # per-fund realised return
        eq_sleeve_ret = float(np.dot(fw, fund_rets))
        debt_ret      = _DEBT_ANNUAL_RETURNS.get(year, DEBT_RET)
        gold_ret      = _GOLD_ANNUAL_RETURNS.get(year, GOLD_RET)
        cash_ret      = _CASH_ANNUAL_RETURNS.get(year, CASH_RET)

        port_ret = (E * eq_sleeve_ret +
                    D * debt_ret +
                    G * gold_ret +
                    C * cash_ret)

        portfolio_value *= (1.0 + port_ret)
        value_series.append(round(portfolio_value, 4))
        annual_ret_series.append(round(port_ret, 6))

        per_year.append({
            "year":         year,
            "pe":           pe,
            "pb":           pb,
            "equity":       round(E, 4),
            "debt":         round(D, 4),
            "gold":         round(G, 4),
            "cash":         round(C, 4),
            "nifty_ret":    round(nifty_ret, 4),
            "port_ret":     round(port_ret, 6),
            "port_value":   round(portfolio_value, 4),
        })

    # ── Summary metrics ──────────────────────────────────────────────────────
    rets_arr  = np.array(annual_ret_series)
    cagr      = float((portfolio_value / 100.0) ** (1.0 / n_years) - 1.0)
    vol_bt    = float(np.std(rets_arr, ddof=1))
    sharpe_bt = float((cagr - RF) / vol_bt) if vol_bt > 0 else 0.0

    # Max drawdown from peak
    peak = 100.0
    max_dd = 0.0
    for v in value_series[1:]:
        if v > peak:
            peak = v
        dd = (v - peak) / peak
        if dd < max_dd:
            max_dd = dd

    return {
        "years":            [y for y in years_sorted],
        "portfolio_value":  value_series[1:],   # year-end values (excl. initial 100)
        "initial_value":    100.0,
        "annual_returns":   annual_ret_series,
        "cagr":             round(cagr, 6),
        "max_drawdown":     round(max_dd, 6),
        "volatility":       round(vol_bt, 6),
        "sharpe":           round(sharpe_bt, 4),
        "per_year":         per_year,
    }


# ── PUBLIC API ───────────────────────────────────────────────────────────────

def get_optimizer_state(funds=None, ann_returns=None, cov_mat=None,
                         vols=None, fund_sharpes=None):
    """
    Returns the full optimizer state dict (parameters + synthetic data).
    Called by app.py on startup if live NAV data is not available.
    """
    if funds is None:
        funds         = FUNDS_DEFAULT
        ann_returns   = _SYNTHETIC_RETURNS
        cov_mat       = _SYNTHETIC_COV
        vols          = _SYNTHETIC_VOLS
        fund_sharpes  = (ann_returns - RF) / vols

    return {
        "funds":        funds,
        "ann_returns":  ann_returns,
        "cov_mat":      cov_mat,
        "vols":         vols,
        "fund_sharpes": fund_sharpes,
    }
