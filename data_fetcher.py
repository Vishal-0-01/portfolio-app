"""
data_fetcher.py — NAV Data Fetching & Preprocessing (FIXED)
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

MIN_HISTORY_DAYS = 750  # ~3 years


def fetch_nav_data(schemes=None) -> pd.DataFrame:
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
                failed.append(name)
                continue

            df = pd.DataFrame(data["data"])
            df["date"] = pd.to_datetime(df["date"], dayfirst=True)
            df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
            df = df.dropna(subset=["nav"])
            df = df[["date", "nav"]].rename(columns={"nav": name})

            all_data = df if all_data.empty else pd.merge(all_data, df, on="date", how="outer")

        except Exception as e:
            logger.warning("Failed to fetch %s: %s", name, e)
            failed.append(name)

    if all_data.empty:
        logger.error("All fetches failed — using synthetic data")
        return None

    if failed:
        logger.info("Skipped funds: %s", failed)

    all_data = all_data.sort_values("date").reset_index(drop=True)

    # ── KEY FIX START ─────────────────────────────────────────────

    df = all_data.set_index("date")

    # 1. Drop funds with insufficient history
    valid_funds = []
    for col in df.columns:
        if df[col].count() >= MIN_HISTORY_DAYS:
            valid_funds.append(col)
        else:
            logger.info("Dropping %s due to insufficient history", col)

    df = df[valid_funds]

    # 2. Forward fill small gaps
    df = df.ffill(limit=3)

    # 3. Drop rows where most funds missing (instead of ALL)
    df = df.dropna(thresh=int(0.8 * len(df.columns)))

    # 4. Final clean
    df = df.dropna()

    # ── KEY FIX END ─────────────────────────────────────────────

    returns = df.pct_change().dropna()

    logger.info(
        "Final dataset: %s funds | %s rows (%s → %s)",
        len(df.columns),
        len(returns),
        returns.index[0].date(),
        returns.index[-1].date()
    )

    return returns


def get_returns(schemes=None):
    returns = fetch_nav_data(schemes)
    if returns is not None and len(returns) > 200:
        return returns, "live"

    logger.info("Using synthetic return data")
    return _synthetic_returns(), "synthetic"


def _synthetic_returns() -> pd.DataFrame:
    from optimizer import FUNDS_DEFAULT, _SYNTHETIC_RETURNS, _SYNTHETIC_COV

    n_days = 1260
    np.random.seed(99)

    try:
        L = np.linalg.cholesky(_SYNTHETIC_COV / 252)
    except np.linalg.LinAlgError:
        cov_jitter = _SYNTHETIC_COV / 252 + 1e-8 * np.eye(len(FUNDS_DEFAULT))
        L = np.linalg.cholesky(cov_jitter)

    Z = np.random.randn(n_days, len(FUNDS_DEFAULT))
    daily = (Z @ L.T) + _SYNTHETIC_RETURNS / 252

    dates = pd.bdate_range(end="2024-12-31", periods=n_days)
    returns = pd.DataFrame(daily, index=dates, columns=FUNDS_DEFAULT)

    return returns
