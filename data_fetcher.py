"""
data_fetcher.py — NAV Data Fetching & Preprocessing
=====================================================
Handles live data fetch via mftool, with graceful fallback to
synthetic data when network is unavailable.
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


def fetch_nav_data(schemes=None) -> pd.DataFrame:
    """
    Fetch historical NAV for all selected schemes.
    Returns daily returns DataFrame aligned on common date.
    Falls back to synthetic data on any failure.
    """
    if schemes is None:
        schemes = SELECTED_SCHEMES

    try:
        from mftool import Mftool
        mf = Mftool()
    except ImportError:
        logger.warning("mftool not installed — using synthetic data")
        return None

    all_data = pd.DataFrame()
    failed = []

    for name, code in schemes.items():
        try:
            data = mf.get_scheme_historical_nav(code)
            if data is None or "data" not in data:
                logger.warning("No data for %s (code %s)", name, code)
                failed.append(name)
                continue

            df = pd.DataFrame(data["data"])
            df["date"] = pd.to_datetime(df["date"], dayfirst=True)
            df["nav"]  = pd.to_numeric(df["nav"], errors="coerce")
            df = df.dropna(subset=["nav"])
            df = df[["date", "nav"]].rename(columns={"nav": name})

            all_data = (df if all_data.empty
                        else pd.merge(all_data, df, on="date", how="outer"))

        except Exception as e:
            logger.warning("Failed to fetch %s: %s", name, e)
            failed.append(name)

    if all_data.empty:
        logger.error("All fetches failed — using synthetic data")
        return None

    if failed:
        logger.info("Skipped funds: %s", failed)

    all_data = all_data.sort_values("date").reset_index(drop=True)

    # Align on common date window to maximise history
    fund_cols = [c for c in all_data.columns if c != "date"]
    idx_data  = all_data.set_index("date")[fund_cols]

    first_valids = {c: idx_data[c].first_valid_index() for c in fund_cols}
    common_start = max(v for v in first_valids.values() if v is not None)
    logger.info("Common start date: %s", common_start.date())

    trimmed = idx_data.loc[common_start:].copy()

    # Forward-fill short gaps (weekends/holidays), then drop remaining NaN rows
    trimmed = trimmed.ffill(limit=3).dropna()

    returns = trimmed.pct_change().dropna()
    logger.info("Returns shape: %s  (%s → %s)",
                returns.shape, returns.index[0].date(), returns.index[-1].date())

    return returns


def get_returns(schemes=None):
    """
    High-level entry. Returns (returns_df, source) where source is
    'live' or 'synthetic'.
    """
    returns = fetch_nav_data(schemes)
    if returns is not None and len(returns) > 100:
        return returns, "live"

    logger.info("Using synthetic return data")
    return _synthetic_returns(), "synthetic"


def _synthetic_returns() -> pd.DataFrame:
    """
    Generate a synthetic daily returns DataFrame that matches the
    statistical properties used in optimizer.py (same seed, same params).
    Used for demo/testing when mftool is unavailable.
    """
    from optimizer import FUNDS_DEFAULT, _SYNTHETIC_RETURNS, _SYNTHETIC_COV

    n_days = 1260  # ~5 years of trading days
    np.random.seed(99)

    # Cholesky decomposition for correlated returns
    try:
        L = np.linalg.cholesky(_SYNTHETIC_COV / 252)
    except np.linalg.LinAlgError:
        # Add small jitter for numerical stability
        cov_jitter = _SYNTHETIC_COV / 252 + 1e-8 * np.eye(len(FUNDS_DEFAULT))
        L = np.linalg.cholesky(cov_jitter)

    Z      = np.random.randn(n_days, len(FUNDS_DEFAULT))
    daily  = (Z @ L.T) + _SYNTHETIC_RETURNS / 252

    import pandas as pd
    dates   = pd.bdate_range(end="2024-12-31", periods=n_days)
    returns = pd.DataFrame(daily, index=dates, columns=FUNDS_DEFAULT)
    return returns
