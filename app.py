"""
app.py — Flask API Server
==========================
Run:  python app.py
API:  http://localhost:5000

Endpoints:
  GET /api/optimize?pe=22&pb=3.5
  GET /api/backtest
  GET /api/frontier
  GET /api/funds
  GET /health
"""

import os
import sys
import json
import logging
import time
from functools import lru_cache

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# ── Path setup ───────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR  = os.path.join(BASE_DIR, "backend")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
sys.path.insert(0, BACKEND_DIR)

from optimizer import (
    optimize_for_pe_pb, compute_frontier, run_backtest,
    run_performance_backtest,
    build_cov_from_returns, get_optimizer_state,
    VOL_CAP, RF,
    NIFTY_MEAN_PE, NIFTY_STD_PE, NIFTY_MEAN_PB, NIFTY_STD_PB,
)
from data_fetcher import get_returns

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("app")

# ── Flask setup ──────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=FRONTEND_DIR)
CORS(app)   # allow browser fetch from any origin during dev

# ── Global optimizer state (loaded once on startup) ──────────────────────────
_STATE = {}
  from data_fetcher import load_data  # or whatever your function name is

  try:
      data = load_data()

      _STATE["funds"] = data["funds"]
      _STATE["ann_returns"] = data["ann_returns"]
      _STATE["cov_matrix"] = data["cov_matrix"]

      print("✅ Data loaded successfully")

  except Exception as e:
      print("❌ Data loading failed:", str(e))
def _init_state():
    global _STATE
    logger.info("Initialising optimizer state...")
    t0 = time.time()

    returns_df, source = get_returns()
    logger.info("Data source: %s  |  shape: %s", source, returns_df.shape)

    funds = list(returns_df.columns)
    ann_returns, cov_mat, vols, fund_sharpes = build_cov_from_returns(returns_df)

    state = get_optimizer_state(funds, ann_returns, cov_mat, vols, fund_sharpes)
    state["source"]   = source
    state["frontier"]            = compute_frontier(ann_returns, cov_mat)
    state["backtest"]            = run_backtest(ann_returns, cov_mat, vols, fund_sharpes, funds)
    state["backtest_performance"] = run_performance_backtest(ann_returns, cov_mat, vols, fund_sharpes, funds)

    logger.info("State ready in %.1fs — %d funds, %d backtest years",
                time.time() - t0, len(funds), len(state["backtest"]))
    _STATE = state


# ── Response helpers ─────────────────────────────────────────────────────────

def _ok(data):
    return jsonify({"status": "ok", "data": data})


def _err(msg, code=400):
    return jsonify({"status": "error", "message": msg}), code


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    return _ok({"ready": bool(_STATE), "source": _STATE.get("source", "uninitialised")})


@app.route("/api/funds")
def api_funds():
    """Return fund list with annualised metrics."""
    funds = _STATE["funds"]
    ann_r = _STATE["ann_returns"].tolist()
    vols  = _STATE["vols"].tolist()
    sharpes = _STATE["fund_sharpes"].tolist()

    return _ok({
        "funds": [
            {
                "name":   funds[i],
                "return": round(ann_r[i], 4),
                "vol":    round(vols[i], 4),
                "sharpe": round(sharpes[i], 3),
            }
            for i in range(len(funds))
        ],
        "source": _STATE["source"],
    })


@app.route("/api/optimize")
def api_optimize():
    """
    GET /api/optimize?pe=22&pb=3.5

    Returns full portfolio allocation + metrics.
    """
    try:
        pe = float(request.args.get("pe", 22))
        pb = float(request.args.get("pb", 3.5))
    except ValueError:
        return _err("pe and pb must be numeric")

    if not (10 <= pe <= 45):
        return _err("pe must be between 10 and 45")
    if not (1.0 <= pb <= 7.0):
        return _err("pb must be between 1.0 and 7.0")

    try:
        result = optimize_for_pe_pb(
            pe, pb,
            _STATE["ann_returns"],
            _STATE["cov_mat"],
            _STATE["vols"],
            _STATE["fund_sharpes"],
            _STATE["funds"],
        )
        result["fund_names"] = _STATE["funds"]
        return _ok(result)

    except Exception as e:
        logger.exception("Optimize failed for PE=%.1f PB=%.1f", pe, pb)
        return _err(f"Optimization error: {str(e)}", 500)


@app.route("/api/backtest")
def api_backtest():
    """GET /api/backtest — return pre-computed 10-year allocation backtest."""
    return _ok(_STATE.get("backtest", []))


@app.route("/api/backtest-performance")
def api_backtest_performance():
    """
    GET /api/backtest-performance
    Returns realised portfolio value series + risk metrics over 2015–2024.
    """
    return _ok(_STATE.get("backtest_performance", {}))


@app.route("/api/frontier")
def api_frontier():
    """GET /api/frontier — return efficient frontier points."""
    return _ok(_STATE.get("frontier", []))


@app.route("/api/meta")
def api_meta():
    """Return global config used by frontend."""
    return _ok({
        "vol_cap":       VOL_CAP,
        "rf":            RF,
        "nifty_mean_pe": NIFTY_MEAN_PE,
        "nifty_std_pe":  NIFTY_STD_PE,
        "nifty_mean_pb": NIFTY_MEAN_PB,
        "nifty_std_pb":  NIFTY_STD_PB,
    })


# ── Serve frontend ────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(FRONTEND_DIR, path)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _init_state()
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    logger.info("Starting server on http://0.0.0.0:%d  (debug=%s)", port, debug)
    app.run(host="0.0.0.0", port=port, debug=debug)
