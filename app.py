"""
app.py — Flask API Server (FIXED)
"""

import os
import sys
import logging
import time

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

#── Path setup ─────────────────────────────────────────────

BASE_DIR     = os.path.dirname(os.path.abspath(file))
BACKEND_DIR  = os.path.join(BASE_DIR, "backend")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
sys.path.insert(0, BACKEND_DIR)

#── Imports ────────────────────────────────────────────────

from optimizer import (
optimize_for_pe_pb, compute_frontier, run_backtest,
run_performance_backtest,
build_cov_from_returns, get_optimizer_state,
VOL_CAP, RF,
NIFTY_MEAN_PE, NIFTY_STD_PE, NIFTY_MEAN_PB, NIFTY_STD_PB,
)
from data_fetcher import get_returns

#── Logging ────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

#── Flask setup ────────────────────────────────────────────

app = Flask(name, static_folder=FRONTEND_DIR)
CORS(app)

#── Global state ───────────────────────────────────────────

_STATE = {}

def _init_state():
global _STATE
logger.info("Initializing optimizer state...")

try:
    returns_df, source = get_returns()

    funds = list(returns_df.columns)
    ann_returns, cov_mat, vols, fund_sharpes = build_cov_from_returns(returns_df)

    state = get_optimizer_state(funds, ann_returns, cov_mat, vols, fund_sharpes)

    state["source"] = source
    state["ann_returns"] = ann_returns
    state["cov_mat"] = cov_mat
    state["vols"] = vols
    state["fund_sharpes"] = fund_sharpes
    state["funds"] = funds

    state["frontier"] = compute_frontier(ann_returns, cov_mat)
    state["backtest"] = run_backtest(ann_returns, cov_mat, vols, fund_sharpes, funds)
    state["backtest_performance"] = run_performance_backtest(
        ann_returns, cov_mat, vols, fund_sharpes, funds
    )

    _STATE = state
    logger.info("State initialized successfully")

except Exception as e:
    logger.error("STATE INIT FAILED: %s", str(e))
    _STATE = {}  # prevent crash

#── Helpers ────────────────────────────────────────────────

def _ok(data):
return jsonify({"status": "ok", "data": data})

def _err(msg, code=400):
return jsonify({"status": "error", "message": msg}), code

#── Routes ────────────────────────────────────────────────

@app.route("/health")
def health():
return _ok({"ready": bool(_STATE)})

@app.route("/api/funds")
def api_funds():
if not _STATE:
return _err("State not initialized", 500)

return _ok({
    "funds": [
        {
            "name": _STATE["funds"][i],
            "return": float(_STATE["ann_returns"][i]),
            "vol": float(_STATE["vols"][i]),
            "sharpe": float(_STATE["fund_sharpes"][i]),
        }
        for i in range(len(_STATE["funds"]))
    ],
    "source": _STATE.get("source", "unknown"),
})

@app.route("/api/optimize")
def api_optimize():
if not _STATE:
return _err("State not initialized", 500)

try:
    pe = float(request.args.get("pe", 22))
    pb = float(request.args.get("pb", 3.5))
except:
    return _err("Invalid PE/PB")

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
    logger.error("Optimization failed: %s", str(e))
    return _err(str(e), 500)

@app.route("/api/backtest")
def api_backtest():
return _ok(_STATE.get("backtest", []))

@app.route("/api/backtest-performance")
def api_backtest_performance():
return _ok(_STATE.get("backtest_performance", {}))

@app.route("/api/frontier")
def api_frontier():
return _ok(_STATE.get("frontier", []))

@app.route("/api/meta")
def api_meta():
return _ok({
"vol_cap": VOL_CAP,
"rf": RF,
"nifty_mean_pe": NIFTY_MEAN_PE,
"nifty_std_pe": NIFTY_STD_PE,
"nifty_mean_pb": NIFTY_MEAN_PB,
"nifty_std_pb": NIFTY_STD_PB,
})

#── Frontend ───────────────────────────────────────────────

@app.route("/")
def index():
return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/"path:path" (path:path)")
def static_files(path):
return send_from_directory(FRONTEND_DIR, path)


if name == "main":
_init_state()
port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port)
