"""
data_fetcher.py — NAV Data Fetching & Preprocessing
=====================================================

ROOT CAUSE OF INFLATED RETURNS (fixed here)
--------------------------------------------

Bug 1 — Window selection bias  [PRIMARY CAUSE]
  Old code: common_start = max(first_valid_index across all funds)
  Effect: Newer funds (Bank of India, WhiteOak launched Jul-Sep 2022)
  forced the entire dataset to start Sep 2022. The window Sep 2022 → Apr 2026
  is a strong bull run (Nifty TRI +85%). Many flexi-cap funds genuinely
  delivered 25-35% CAGR in that window. Not a math error — a data selection
  error. The optimizer was given an unrepresentative short sample.

Bug 2 — Arithmetic mean annualization overstates CAGR  [SECONDARY CAUSE]
  Old code: mean_ret = returns_df.mean() * 252   ← arithmetic mean
  Arithmetic annualized return > geometric CAGR by approximately vol²/2.
  For vol=0.20: overstatement = 2% per year.
  For vol=0.25: overstatement = 3.1% per year.
  This compounds on top of Bug 1.

Fix
---
  1. Per-fund independent history: each fund uses its own full date range.
     No forced common start. Covariance uses pairwise overlapping periods.
  2. Geometric CAGR via log-return method:
       ann_ret = exp(mean(log(1 + r_daily)) * 252) - 1
     This gives the true compounded annual growth rate.
  3. Minimum 3-year (756 trading day) history requirement. Funds with
     less are excluded to avoid recency/bull-market bias entirely.
  4. Max 5-year (1260 trading day) lookback to keep data relevant.
  5. Weekday-only filter before pct_change to avoid any weekend artifacts.
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

SELECTED_SCHEMES = {
    "Parag Parikh":  "122639",
    "Quant":         "120843",
    "ICICI":         "134799",
    "Kotak":         "145552",
    "Motilal Oswal": "129046",
    "HDFC":          "118955",
    "Aditya Birla":  "120564",
    "ITI":           "151379",
    "Tata":          "144546",
    "Invesco":       "149763",
    "Bank of India": "148404",
    "HSBC":          "120046",
    "Edelweiss":     "140353",
    "WhiteOak":      "150346",
}

# Minimum trading days of history required to include a fund (~3 years).
# Funds below this threshold are excluded to prevent short-window bias.
MIN_TRADING_DAYS = 756

# Maximum lookback in trading days (~5 years). Uses the most recent N days.
# None = use all available history.
MAX_LOOKBACK_DAYS = 1260


def _fetch_single_fund(mf, name: str, code: str):
    """
    Fetch NAV series for one fund. Returns pd.Series (date index, NAV values)
    sorted ascending on trading days only. Returns None on any failure.
    """
    try:
        data = mf.get_scheme_historical_nav(code)
        if data is None or "data" not in data or not data["data"]:
            logger.warning("No data for %s (code %s)", name, code)
            return None

        df = pd.DataFrame(data["data"])

        if "date" not in df.columns or "nav" not in df.columns:
            logger.warning("Unexpected columns for %s: %s", name, df.columns.tolist())
            return None

        df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
        df["nav"]  = pd.to_numeric(df["nav"], errors="coerce")
        df = df.dropna(subset=["date", "nav"])

        if df.empty:
            return None

        # Sort ascending (mftool returns newest-first)
        df = df.sort_values("date").reset_index(drop=True)

        # Keep weekdays only — remove any stray weekend entries
        df = df[df["date"].dt.dayofweek < 5]

        # Remove duplicate dates (keep last/most recent entry for that day)
        df = df.drop_duplicates(subset="date", keep="last")

        return df.set_index("date")["nav"].rename(name)

    except Exception as e:
        logger.warning("Failed to fetch %s: %s", name, e)
        return None


def fetch_nav_data(schemes=None):
    """
    Fetch NAV for all schemes. Returns DataFrame of DAILY RETURNS (not NAVs).

    Each fund is processed independently over its own date range.
    The returned DataFrame has NaN where a fund has no data for a given date.
    The covariance matrix in optimizer.py uses pairwise computation which
    handles these NaNs correctly (overlapping periods per pair).

    Returns None if fewer than 3 funds can be loaded.
    """
    if schemes is None:
        schemes = SELECTED_SCHEMES

    try:
        from mftool import Mftool
        mf = Mftool()
    except ImportError:
        logger.warning("mftool not installed — using synthetic data")
        return None

    nav_series     = {}
    failed         = []
    excluded_short = []

    for name, code in schemes.items():
        s = _fetch_single_fund(mf, name, code)
        if s is None:
            failed.append(name)
            continue

        # Apply max lookback (most recent N trading days)
        if MAX_LOOKBACK_DAYS is not None and len(s) > MAX_LOOKBACK_DAYS:
            s = s.iloc[-MAX_LOOKBACK_DAYS:]

        # Enforce minimum history requirement
        if len(s) < MIN_TRADING_DAYS:
            excluded_short.append((name, len(s)))
            logger.warning(
                "Excluding %s: only %d days of history (need %d)",
                name, len(s), MIN_TRADING_DAYS
            )
            continue

        nav_series[name] = s
        logger.info(
            "Loaded %-18s  %4d days  %s → %s  NAV %6.2f → %7.2f",
            name, len(s),
            s.index[0].date(), s.index[-1].date(),
            s.iloc[0], s.iloc[-1],
        )

    if failed:
        logger.info("Fetch failed: %s", failed)
    if excluded_short:
        logger.info("Excluded (history too short): %s", excluded_short)

    if not nav_series:
        logger.error("No funds loaded successfully")
        return None

    # Combine: outer join preserves each fund's full date range
    all_nav = pd.concat(nav_series.values(), axis=1, join="outer")
    all_nav.sort_index(inplace=True)
    all_nav.index.name = "date"

    # Forward-fill small gaps (public holidays, occasional missing NAV
    # submissions). Limit = 3 prevents manufacturing multi-day returns.
    all_nav = all_nav.ffill(limit=3)

    # Daily returns
    returns = all_nav.pct_change().iloc[1:]   # drop the first all-NaN row
    returns = returns.dropna(how="all")        # drop rows where every fund is NaN

    logger.info(
        "Returns: %d rows x %d funds  (%s → %s)",
        len(returns), len(returns.columns),
        returns.index[0].date(), returns.index[-1].date(),
    )
    return returns


def get_returns(schemes=None):
    """
    Returns (returns_df, source) where source is 'live' or 'synthetic'.
    """
    returns = fetch_nav_data(schemes)
    if returns is not None and len(returns.columns) >= 3:
        return returns, "live"

    logger.info("Falling back to synthetic return data")
    return _synthetic_returns(), "synthetic"


def _synthetic_returns():
    """
    Synthetic daily returns: ~13% geometric CAGR, 17-21% annualized vol.
    Used only when mftool is unavailable.
    """
    from optimizer import FUNDS_DEFAULT, _SYNTHETIC_RETURNS, _SYNTHETIC_COV

    n_days = 1260
    np.random.seed(99)

    try:
        L = np.linalg.cholesky(_SYNTHETIC_COV / 252)
    except np.linalg.LinAlgError:
        cov_j = _SYNTHETIC_COV / 252 + 1e-8 * np.eye(len(FUNDS_DEFAULT))
        L = np.linalg.cholesky(cov_j)

    Z     = np.random.randn(n_days, len(FUNDS_DEFAULT))
    daily = Z @ L.T + _SYNTHETIC_RETURNS / 252

    dates   = pd.bdate_range(end="2024-12-31", periods=n_days)
    returns = pd.DataFrame(daily, index=dates, columns=FUNDS_DEFAULT)
    return returns
